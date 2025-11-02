import os 
import hashlib 
import asyncio
import zipfile
import shutil
from pathlib import Path
from sseqs import msa
from fastapi import Request, Form, FastAPI
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, FileResponse, HTMLResponse
import uvicorn
import io, tarfile
import argparse

# parse command line arguments
parser = argparse.ArgumentParser(description="Run MSA Server")
parser.add_argument("-port", '-p', default=8000, type=int, help="Port for server. ")
parser.add_argument("-msa_num", default=4096, type=int, help='Max number of lines in MSA')
parser.add_argument("-msa_fft_rank", default=4, type=int, help='Larger is more accurate/slower. ')
parser.add_argument("-msa_top_fft", default=200, type=int, help='')
parser.add_argument("-msa_top_sw", default=10, type=int, help='')
parser.add_argument("-msa_top_sw_affine", default=2, type=int, help='')
args, _ = parser.parse_known_args()

# setup folder structure/app/env variables
os.makedirs('requests/', exist_ok=True)
chunks = int(os.environ.get('CHUNKS', 4))
app = FastAPI()
prediction_lock = asyncio.Lock()
msa_lock = asyncio.Lock()
log: str = ""

# Capture stdout/stderr to log
import sys
class TeeStream:
    def __init__(self, original_stream):
        self.original_stream = original_stream
    
    def write(self, text):
        global log
        self.original_stream.write(text)
        self.original_stream.flush()
        log += text
        # Keep only last ~50000 chars to prevent unbounded growth
        if len(log) > 50000:
            log = log[-50000:]
    
    def flush(self):
        self.original_stream.flush()
    
    def isatty(self):
        return self.original_stream.isatty()
    
    def fileno(self):
        return self.original_stream.fileno()

sys.stdout = TeeStream(sys.stdout)
sys.stderr = TeeStream(sys.stderr)

# Test that logging is working
print("=== Server logging initialized ====")

# accept {.fasta, .yaml} files, write to disk, then call boltz2. 
@app.post("/boltz")
async def upload(files: list[UploadFile] | None = File(None), args: str = Form("")):
    if not files: return {"error": "no file"}
    
    # Check if GPU or MSA is busy
    if prediction_lock.locked():
        return {"error": "gpu busy wait"}
    if msa_lock.locked():
        return {"error": "msa busy wait"}
    
    # Validate args: only allow alphanumeric, spaces, hyphens, underscores
    if args and not all(c.isalnum() or c in ' -_' for c in args):
        return {"error": "invalid args"}
    
    # Create deterministic hash from file contents and args
    # Read all files once and sort by content for determinism
    file_data = [await file.read() for file in files]
    file_data.sort()
    combined_content = b'@'.join(file_data)
    
    # Include args in cache key only if non-empty
    if args.strip():
        combined_content += b'@ARGS@' + args.encode('utf-8')
    
    cache_key = hashlib.sha256(combined_content).hexdigest()
    cache_dir = Path('cache_predictions')
    cache_dir.mkdir(exist_ok=True)

    # Check if we have cached results (filename format: {request_id}_{cache_key}.zip)
    cached_files = list(cache_dir.glob(f'*_{cache_key}.zip'))
    if cached_files:
        # Extract request ID from filename
        request_id = cached_files[0].stem.split('_')[0]
        return {"status": "cached", "request_id": request_id}
    
    # No cache, proceed with prediction (acquire lock)
    async with prediction_lock:
        num = len(os.listdir('requests/')) + 1
        os.makedirs(f'requests/{num}/fastas/', exist_ok=True)

        # Use the already-read content from file_data (no need to re-read)
        for i, content in enumerate(file_data):
            file_path = f'requests/{num}/fastas/{i}.fasta'
            with open(file_path, 'wb') as f:
                f.write(content)
        
        # Run boltz and wait for completion, capturing output
        # Build command with optional args
        cmd = f"boltz predict requests/{num}/fastas/  --output_format pdb --use_msa_server --msa_server_url http://localhost:8000 --out_dir requests/{num}/"
        if args.strip():
            cmd += f" {args}"
        
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        
        # Stream output line by line so it gets captured by our TeeStream
        async for line in process.stdout:
            print(line.decode('utf-8', errors='replace'), end='')
        
        await process.wait()
        
        # Zip the output directory
        out_dir = Path(f'requests/{num}/')
        zip_path = f'requests/{num}/predictions.zip'
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in out_dir.rglob('*'):
                if file_path.is_file() and file_path.name != 'predictions.zip':
                    arcname = file_path.relative_to(out_dir)
                    zinfo = zipfile.ZipInfo(str(arcname))
                    zinfo.date_time = (1980, 1, 1, 0, 0, 0)
                    with open(file_path, 'rb') as f:
                        zipf.writestr(zinfo, f.read())
        
        # Cache the result with request ID in filename for future lookup
        cached_zip = cache_dir / f'{num}_{cache_key}.zip'
        shutil.copy2(zip_path, cached_zip)
        
        return {"status": "completed", "request_id": str(num)}


# code below supports `boltz predict --msa_server_url=0.0.0.0:8000 --use_msa_server`
result_store: dict[str, Path] = {}
@app.post("/ticket/msa", include_in_schema=False)
@app.post("/ticket/pair", include_in_schema=False)
async def ticket_msa(request: Request, q: str = Form(...)):
    # Check if MSA is busy
    if msa_lock.locked():
        return Response(status_code=503, content="msa busy - please wait a bit")
    
    q = str(q).split('\n')

    # cache results in folder specific to options. 
    cache_folder = f"{chunks}_{args.msa_num}_{args.msa_top_fft}_{args.msa_fft_rank}_{args.msa_top_sw}_{args.msa_top_sw_affine}"
    os.makedirs(f"cache_msa/{cache_folder}/", exist_ok=True)

    # fetch all proteins from input 
    heads, proteins = q[::2], q[1::2]
    out_paths = []
    for protein in proteins: 
        protein_hash = hashlib.sha256(protein.encode('utf-8')).hexdigest()
        out_paths.append(f"cache_msa/{cache_folder}/{protein_hash}.a3m")

    # compute unique proteins and output paths 
    unique_proteins, a3m_paths = [], []
    for protein in list(set(proteins)):
        protein_hash = hashlib.sha256(protein.encode('utf-8')).hexdigest()
        a3m_path = f"cache_msa/{cache_folder}/{protein_hash}.a3m" 
        if os.path.exists(a3m_path): continue
        unique_proteins.append(protein)
        a3m_paths.append(a3m_path)

    # compute MSA for all proteins (with lock)
    if len(unique_proteins)>0:
        async with msa_lock:
            msa(unique_proteins, 
                    a3m_paths,
                    fft_rank=args.msa_fft_rank,
                    top_fft=args.msa_top_fft, 
                    top_sw=args.msa_top_sw, 
                    top_sw_affine=args.msa_top_sw_affine, 
                    num_msas=args.msa_num,
                    bs=100_000_000)

    # fix header in output format. 
    for i, a3m_path in enumerate(out_paths): 
        a = open(a3m_path, 'r').read().replace("@",'X')
        lines = a.split('\n')
        lines[0] = q[i*2] 
        open(a3m_path, 'w').write("\n".join(lines))

    # add seperator (?null ascii character?). 
    uniref_content = b"\x00\n".join(Path(p).read_bytes() for p in out_paths)

    # satisfy the output format
    tar_id = protein_hash 
    tar_path = Path(tar_id.replace('.a3m','_tar'))
    tar_path.mkdir(exist_ok=True)
    tar_file = tar_path / f"{tar_id}.tar.gz"

    with tarfile.open(tar_file, "w:gz") as tar:
        is_pair = request.url.path.endswith("/ticket/pair")
        name = "pair.a3m" if is_pair else "uniref.a3m"
        ti = tarfile.TarInfo(name=name)
        ti.size = len(uniref_content)
        tar.addfile(ti, io.BytesIO(uniref_content))
        env_name = "bfd.mgnify30.metaeuk30.smag30.a3m"
        ti2 = tarfile.TarInfo(name=env_name)
        ti2.size = 0
        tar.addfile(ti2, io.BytesIO(b""))

    result_store[tar_id] = tar_file

    return {"status": "COMPLETE", "id": tar_id}

# Endpoint to serve the tar.gz back to run_mmseqs2
@app.get("/result/download/{tar_id}", include_in_schema=False)
async def result_download(tar_id: str):
    path = result_store.get(tar_id)
    if path is None or not path.exists():
        return Response(status_code=404)
    return FileResponse(path, media_type="application/gzip", filename="out.tar.gz")

# don't list files that didn't finish := "does it have a .pdb file" 
@app.get("/requests")
async def list_requests():
    import datetime
    requests = []
    for d in Path('requests/').iterdir():
        if d.is_dir():
            # Check if there are any PDB files in the directory
            pdb_files = list(d.rglob('*.pdb'))
            if not pdb_files:
                continue  # Skip if no PDB files found
            
            mtime = d.stat().st_mtime
            date_str = datetime.datetime.fromtimestamp(mtime).strftime('%m-%d %H:%M')
            requests.append({'id': d.name, 'date': date_str, 'timestamp': mtime})
    return requests

@app.get("/requests/{request_id}")
async def download_request(request_id: str):
    folder = Path(f'requests/{request_id}')
    if not folder.exists() or not folder.is_dir():
        return Response(status_code=404)
    zip_path = folder / 'predictions.zip'
    if zip_path.exists():
        return FileResponse(zip_path, media_type="application/zip", filename=f"predictions_{request_id}.zip")
    return Response(status_code=404)

@app.get("/status")
async def get_status():
    # Get last 100 lines of log
    lines = log.split('\n')
    last_100_lines = '\n'.join(lines[-100:])
    
    return {
        "prediction_lock": "busy" if prediction_lock.locked() else "free",
        "msa_lock": "busy" if msa_lock.locked() else "free",
        "log": last_100_lines
    }

@app.get("/", response_class=HTMLResponse)
async def protboard():
    return open('protboard.html', 'r').read() 

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=args.port, reload=False, access_log=False)