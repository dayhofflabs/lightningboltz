import os 
import sys 
import hashlib
import pickle 
import numpy as np 
import math 
from torch.utils.cpp_extension import load
import torch as th
from tqdm import tqdm
import psutil
import time 

# Loads `CHUNKS` GB of MSA database to RAM. 
chunks = int(os.environ.get('CHUNKS', 4))
assert chunks > 0

# Look for database. 
xbit_path = os.environ.get('DBPATH', 'uniref_bfd_mgy_cf.xbit')
package_dir = os.path.dirname(os.path.abspath(__file__))

if not os.path.exists(xbit_path):
    print(f"Didn't find MSA database at `{xbit_path}`")
    print("\twget https://foldify.org/uniref_bfd_mgy_cf.xbit")
    print("\texport DBPATH=$PWD/uniref_bfd_mgy_cf.xbit")
    exit()

# @alex:  supports cache loading in ipython notebook
#         reach out if you can find nicer way to achieve this. 
if sys.argv[-1] != '-loaded': 

    # Load CUDA kernels. 
    verbose = True
    xbit = load("xbit", sources=[f"{package_dir}/xbit_bind.cpp",   f"{package_dir}/xbit.cu"],   extra_cuda_cflags=["-O3", "--use_fast_math"], extra_cflags=["-O3"], verbose=verbose)
    conv = load("conv", sources=[f"{package_dir}/filter_bind.cpp", f"{package_dir}/filter.cu"], extra_cuda_cflags=["-O3", "--use_fast_math"], extra_cflags=["-O3"], verbose=verbose)
    sw   = load("sw",   sources=[f"{package_dir}/sw_bind.cpp",     f"{package_dir}/sw.cu"],     extra_cuda_cflags=["-O3", "--use_fast_math"], extra_cflags=["-O3"], verbose=verbose)

    # Read antibody dataset. 
    uniref_bfd_chunks = chunks 
    AB = os.environ.get('AB', False)
    if AB: chunks += 23 

    avail_gb_ram = psutil.virtual_memory().available / 1_000_000_000
    needed_gb_ram = (chunks+5)
    if avail_gb_ram <  needed_gb_ram: 
        print(f"Need {needed_gb_ram}GB RAM, only {avail_gb_ram}GB available. ")
        print(f"You can run with e.g. 7GB data using `CHUNKS=7 python server.py`. ") 
        exit()

    # Prepare pinned RAM to enable fast RAM->VRAM transfer.             ~ 1 GB/s
    pinned_ram = [th.empty((1_000_000_000), dtype=th.uint8, pin_memory=True) for _ in tqdm(range(chunks), desc=f"Preparing {chunks}GB of pinned RAM. ")]

    with open(xbit_path, "rb") as fh:
        # todo:  rewrite to numpy slicing notation, e.g., SLICE=[7:117:7]
        skip = int(os.environ.get('SKIP', 0))
        for i, t in enumerate(tqdm(pinned_ram[:uniref_bfd_chunks], desc=f"Reading {chunks}GB MSA data from `{xbit_path}` into pinned RAM. ")):
            fh.seek( (i+skip) * 1000 * 1000 * 1000)
            fh.readinto(t.numpy())

    # Read antibody dataset. 
    if AB:
        with open(AB, "rb") as fh:
            for i, t in enumerate(tqdm(pinned_ram[-23:], desc=f"Reading {chunks}GB from `antibody.xbit` to pinned RAM")):
                fh.seek(i * 1000 * 1000 * 1000)
                fh.readinto(t.numpy())

    sys.argv.append('-loaded')

from sseqs.to_a3m import to_a3m

import threading
_last={}

def msa(queries, 
        filenames=None, 
        num_msas=4096, 
        save_raw='', 
        verbose=True, 
        bs=50_000_000, 
        fft_rank=4,      # the rank of the SVD(BLOSUM62_SCORE) approximation
        top_fft=200,     # pass on 1/top_fft from fft filter 
        top_sw=10,       # pass on 1/top_sw from sw filter 
        top_sw_affine=2, # pass on 1/top_sw_affine from sw_filter (used to be savetopk)
        return_timings=False,
        sync_time=False, 
        dlength=1e6, 
        gpu: int = 0, 
        blocking: bool = True): 

    # Support 4 gpu reading MSA from same RAM asynchronously = non-blocking
    if not blocking: 
        print(gpu)
        t=_last.get(gpu)
        if t is not None and t.is_alive(): t.join()
        def _target():
            with th.cuda.device(gpu): msa(
                queries, filenames, num_msas, save_raw, 
                verbose, bs, fft_rank, top_fft, top_sw,       
                top_sw_affine, return_timings, sync_time, 
                dlength, gpu, blocking=True)
        nt=threading.Thread(target=_target)
        _last[gpu]=nt
        nt.start()
        return nt

    # support different gpu in blocking/sequential mode. 
    with th.cuda.device(gpu):

        if type(queries) == str: queries = [queries]
        if type(filenames) == str: filenames = [filenames]

        # BLOSUM62 scores used by DFT-low rank-prefilter. 
        BLOSUM62 = np.array([4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0, -1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3, -2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3, -2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3, 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1, -1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2, -1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2, 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3, -2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3, -1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3, -1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1, -1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2, -1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1, -2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1, -1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2, 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2, 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0, -3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3, -2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1, 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4,]).reshape(20,20)

        # Create hash file names if no filename provided. 
        if filenames == None: 
            hashes = [hashlib.sha256(query.encode()).hexdigest()[:8] for query in queries]
            filenames = [hash + '.a3m' for hash in hashes]

        aas = "ARNDCQEGHILKMFPSTWYV"                      # 20 standard AAs
        AA_TO_POS = {aa: i for i, aa in enumerate(aas)}
        AA_TO_POS.update({"X": 20, "-": 20, "*": 20})      # gap / unknown

        # Prefilter is gapless/fft conv alignment with low-rank approx to BLOSUM62 scoring matrix.
        # Found better perf when 
        # 1. bias SVD towards query 
        # 2. for rank=1 works better when we kill negative part 
        if fft_rank == 1: BLOSUM62[BLOSUM62 < 0] = 0 

        _weight_LUTS_As, _weight_LUTS_Bs = [], []
        for query in queries: 
            w = np.sqrt(np.array([query.count(a) for a in aas]) / len(query) )  # row-weights
            U, S, Vt = np.linalg.svd(w[:, None] * BLOSUM62 )     # weighted SVD

            # Build two LUT lists: one for the query (√σ · u) and one for the db seqs (√σ · v)
            _weight_LUTS_A, _weight_LUTS_B = [], []
            for s, u_col, v_row in zip(S, U.T, Vt):
                _weight_LUTS_A.append(
                    th.tensor(np.append(np.sqrt(s) * u_col, 0.0),  device='cuda', dtype=th.float32)# pad for unknown/gap
                )
                _weight_LUTS_B.append(
                    th.tensor(np.append(np.sqrt(s) * v_row, 0.0), device='cuda', dtype=th.float32)
                )
            _weight_LUTS_A = th.vstack(_weight_LUTS_A).contiguous().to(th.float32)
            _weight_LUTS_B = th.vstack(_weight_LUTS_B).contiguous().to(th.float32)
            _weight_LUTS_As.append(_weight_LUTS_A)
            _weight_LUTS_Bs.append(_weight_LUTS_B)
        del _weight_LUTS_A, _weight_LUTS_B, w, U, S, Vt

        max_sw, max_fft, matches = 0, 0, 0 
        seq_matchess = [[] for _ in queries]
        seq_scoress = [[] for _ in queries] 
        seq_scores_ffts = [[] for _ in queries]
        seq_scores_gaps = [[] for _ in queries]
        AA_MAP = {aa: idx for idx, aa in enumerate("ARNDCQEGHILKMFPSTWYV")}# AA_TO_POS?

        query_encs, query_reversed_tensors = [], []
        for query in queries: 
            query_enc = th.tensor([AA_MAP[aa] for aa in query if aa in AA_MAP], dtype=th.uint8, device="cuda")
            query_reversed_tensor = th.tensor([AA_TO_POS.get(c, 20) for c in query[::-1]], dtype=th.uint8, device='cuda')
            query_encs.append(query_enc)
            query_reversed_tensors.append(query_reversed_tensor)
        del query_enc, query_reversed_tensor

        t_dec, t_out_cpu, t_prep1, t_prep2, t_fft, t_sw, t_out, t_log = 0, 0, 0, 0, 0, 0, 0, 0
        t_topk1, t_topk2 = 0, 0
        t_prep_sw = 0
        _score1, _score2 = 0,0
        t_to_vram = 0
        t_prep_fft = 0

        arange = th.arange(int(bs*1.6), device='cuda', dtype=th.int32)

        """
        WARNING:  The following code is complicated -- read below or be lost. 
        Idea:     While RAM->VRAM loads chunk i+1, we align chunk i. 
        Complex:  Any host operation breaks asynchronously loading (=> ~2x slower).
        Example:  tensor[:lengths.max()] makes code 2x slower, because
                behind the scenes tensor[:num] transfers `num` to host with `num.item()`
        """

        # need seperate streams to overlap copying and processing
        copy_stream = th.cuda.Stream(gpu)   # create once at start
        compute_stream = th.cuda.Stream(gpu)   # create once at start

        MAX_GOOD_INDXs = 1000
        preallocated_arange = th.arange(int(MAX_GOOD_INDXs), device='cuda', dtype=th.int32)
        b = th.zeros_like(pinned_ram[0], device='cuda')

        num_seqs = 0
        
        non_blocking = True
        for filename in filenames:
            if os.path.exists(filename): os.remove(filename)

        if verbose: pbar = tqdm(pinned_ram, unit='GB')
        else: pbar = pinned_ram

        for i, a_pin in enumerate(pbar): 
            # Load chunk `i+1` RAM->VRAM while decompressing/convolving batch `i`.
            t0 = time.time()
            with th.cuda.stream(copy_stream):
                if i != 0: 
                    b.copy_(next)  # loading current one last iteration 
                    th.cuda.synchronize()
                    if i+1 < len(pinned_ram): next = pinned_ram[i+1].cuda(non_blocking=non_blocking) # load next without in this iteration withoutblocking
                else: 
                    b.copy_(a_pin.cuda()) 
                    th.cuda.synchronize()
                    if chunks > 1: next = pinned_ram[i+1].cuda(non_blocking=non_blocking)
                    if sync_time: th.cuda.synchronize()
            
            t_to_vram += time.time() - t0
            assert pinned_ram[i].is_pinned()

            with th.cuda.stream(compute_stream):
                batchess = [[] for _ in queries]
            
                assert b.numel() % bs == 0, "b.numel() must be divisible by bs"
                for j in range(0, math.ceil(b.numel()/bs), 1): 
                    # ── Decompress ───────────────────────────────────────────────────────────────
                    t0 = time.time() 
                    ascii = xbit.decompress(b, int(bs*8/5), j*bs) # does b[l*bs:] inside. 
                    if sync_time: th.cuda.synchronize()
                    t_dec += time.time() - t0#
                    # ── Decompress ───────────────────────────────────────────────────────────────

                    # ── Prepare    ───────────────────────────────────────────────────────────────
                    t0 = time.time() # this is 0.56s out of 1.19s for 10chunks
                    delim_pos = arange[ascii==64]
                    if sync_time: th.cuda.synchronize()
                    t_prep1 += time.time() - t0
                    t0 = time.time()
                    #keep_mask = th.cat([ th.ones(1, dtype=th.bool, device=ascii.device), delim_pos[1:] != delim_pos[:-1] + 1 ])
                    #delim_pos = delim_pos[keep_mask]
                    delim_pos = delim_pos.flatten()

                    for qi, query in enumerate(queries): 
                        # sequence starts are the bytes immediately after the delimiter
                        starts = delim_pos + 1
                        starts = starts[starts < ascii.numel()] # drop any trailing '@' 

                        offsets = th.cat([
                            starts.to(th.int32),
                            th.tensor([ascii.numel()], dtype=th.int32, device=ascii.device)  # sentinel
                        ])
                        lengths = offsets - th.concat([th.tensor([0], dtype=th.int32, device=ascii.device), offsets[:-1]])  
                        lengths[:-1] -= 1 # -1 because of delimiter (but not last one no delimiter there. )
                        last_is_delimiter = ascii[-1] == 64
                        if last_is_delimiter:  lengths[-1] -= 1
                        max_seq_length = float(lengths.float().max().item())

                        # assuming https://x.com/saakohl/status/1980206575802269821
                        # => accuracy comes from seqs of similar length -- appears to hold empirically.
                        if abs(len(query) - max_seq_length) > dlength: continue 
                        lengths = lengths[lengths!=0] 
                        num_seqs += lengths.shape[0]
                        if sync_time: th.cuda.synchronize()
                        t_prep2 += time.time() - t0
                        # ── Prepare    ───────────────────────────────────────────────────────────────
                    
                        # ── FFT ───────────────────────────────────────────────────────────────
                        t0 = time.time()
                        n = 1 << (max(len(query) , int(lengths.max())) - 1).bit_length()  # next power-of-2 -- might be able to half! 
                        k = max(1, min(fft_rank, len(_weight_LUTS_As[qi])))
                        scores = th.zeros((len(lengths)), device='cuda', dtype=th.float16) # move this outside -- init once. 

                        maxlen  = lengths.max()
                        _offsets = th.zeros(lengths.size(0) + 1, device='cuda', dtype=th.int32)
                        _offsets[1:] = th.cumsum(lengths+1, 0)

                        if sync_time: th.cuda.synchronize()
                        t_prep_fft += time.time() - t0 
                        t0 = time.time()

                        #H = n >> 1  # n is already a power of 2
                        query_rfft_tensor = th.zeros(k * (n + 1), device='cuda', dtype=th.float16)
                        
                        conv.warp_corr(scores, 
                                        query_reversed_tensors[qi], # new arg: query reversed as uint8 indices
                                        _weight_LUTS_As[qi], # Pass the whole LUTs_A (R, vocab_size_x_float)
                                        _weight_LUTS_Bs[qi],
                                        ascii, 
                                        _offsets[:-1],
                                        lengths,
                                        k, 
                                        len(query), # This is La
                                        int(maxlen), 
                                        n,
                                        query_rfft_tensor) 

                        if sync_time: th.cuda.synchronize()
                        t_fft += time.time() - t0
                        # ── FFT ───────────────────────────────────────────────────────────────

                        # ── TOPK ───────────────────────────────────────────────────────────────
                        # TODO: the last bit is always skipped here when scores.shape[0]<topk
                        # => scores.shape[0]//topk == 0 
                        t0 = time.time()
                        good_idx = th.topk(scores, k=scores.shape[0]//top_fft)[1] # take top 1/1000'th
                        if sync_time: th.cuda.synchronize()
                        t_topk1 += time.time() - t0
                        t0 = time.time()

                        # pass in good_idx and ascii to sw_cuda_blosum62 -- then do 
                        if sync_time: th.cuda.synchronize()
                        t_prep_sw += time.time()-t0
                        t0 = time.time()

                        good_idx = th.sort(good_idx)[0][1:-1] # TODO: start/stop has issue. 
                        if good_idx.shape[0] == 0: 
                            #print(j, good_idx.shape, ascii.shape)
                            continue # no code sequences! 

                        #print(query_encs[qi])
                        sw_scores = sw.sw_cuda_blosum62( 
                                        query_encs[qi], 
                                        good_idx.to(th.int32), 
                                        ascii, 
                                        delim_pos.to(th.int32),
                                        gap_penalty_param=-1 
                                    )

                        # if we select 1/6'th it'd be 2x slower?
                        # is there a bias in th ones that pass through here?
                        # compare time of sw/affine_sw 
                        _topk = th.topk(sw_scores, k=sw_scores.shape[0]//top_sw)  

                        # if we do open_gap for all does it beat the other? 
                        gap_sw_scores = sw.sw_cuda_affine(
                                        query_encs[qi],
                                        good_idx[_topk.indices].to(th.int32),   
                                        ascii,
                                        starts.to(th.int32),                    
                                        gap_open=11, gap_extend=1)

                        # only keep top_sw_affine ?
                        __topk = th.topk(gap_sw_scores, k=top_sw_affine)
                        gap_sw_scores = gap_sw_scores[__topk.indices]
                        _topk = _topk.indices[__topk.indices]   # for later

                        if sync_time: th.cuda.synchronize()
                        t_sw += time.time() - t0
                        # ── SW ───────────────────────────────────────────────────────────────

                        # @alex: check this doesn't skew things -- most batches all have same length tho. 
                        #sw_scores = [u/math.sqrt(len(query)*len(targets[i])) for i, u in enumerate(sw_scores)]
                        gap_sw_scores = gap_sw_scores /(len(query)*maxlen).sqrt() 
                        t0 = time.time()

                        # we can also just do topk here -- then give result straight to to_a3m? 
                        starts = th.cat((th.zeros(1, dtype=lengths.dtype, device='cuda'), 
                                            th.cumsum(lengths[:-1] + th.ones(1, dtype=lengths.dtype, device='cuda'), dim=0)))
                        good_starts, good_stops = starts[good_idx], lengths[good_idx]

                        #scores = scores[good_idx]
                        if top_sw_affine > sw_scores.shape[0]:  
                            #print("I am confusing you")
                            continue 

                        if sync_time: th.cuda.synchronize()
                        t_topk2 += time.time() - t0
                        t0 = time.time()

                        good_idx = _topk#.indices
                        sw_scores, scores = sw_scores[good_idx], scores[good_idx]
                        good_starts, good_stops = good_starts[good_idx], good_stops[good_idx]

                        # asynch way to compute:  _arange = torch.arange(th.max(good_stops), device='cuda')
                        _arange = (preallocated_arange < th.max(good_stops)).nonzero(as_tuple=True)[0]
                        idx  = good_starts[:, None] + _arange
                        current_batch_seqs = ascii.expand(idx.size(0), -1).gather(1, idx)
                        batchess[qi].append((j, i, scores.clone(), sw_scores.clone(), gap_sw_scores.clone(), good_stops.clone(), current_batch_seqs.clone()))
                    #batches.append((good_stops, current_batch_seqs))
                    #print(scores)
                    if sync_time: th.cuda.synchronize()
                    t_out += time.time() - t0
                    del ascii 

                for qi, batches in enumerate(batchess): 
                    for j,i,scores, sw_scores, gap_sw_scores, good_stops, current_batch_seqs in batches: 
                    #for good_stops, current_batch_seqs in batches: 
                        t0 = time.time()
                        # for 50 chunks:  this reduce from 1s to 0.1s
                        B = current_batch_seqs.shape[1]
                        buf_u8 = current_batch_seqs.cpu().numpy()      # (n, B)  dtype=uint8
                        n, B    = buf_u8.shape
                        rows_bytes = buf_u8.view(f'S{B}').ravel()      # (n,)   dtype='|S<B>'
                        rows_unicode = rows_bytes.astype(f'U{B}')      # (n,)   dtype='<U<B>'
                        lengths = good_stops.cpu().tolist()            # Python ints
                        current_batch_seqs = [s[:l] for s, l in zip(rows_unicode, lengths)]

                        if sync_time: th.cuda.synchronize()
                        t_out_cpu += time.time() - t0

                        # --- OUT ---------------------------------------------
                        t0 = time.time()
                        max_sw = max(max_sw, th.max(sw_scores))
                        max_fft = max(max_fft, int(th.max(scores[scores!=th.inf]).item()))
                        matches += len(current_batch_seqs)

                        seq_matchess[qi].append(current_batch_seqs)
                        seq_scoress[qi].append(sw_scores.cpu().numpy())
                        seq_scores_gaps[qi].append(gap_sw_scores.cpu().numpy())
                        seq_scores_ffts[qi].append(scores.cpu().numpy())

                        if sync_time: th.cuda.synchronize()
                        t_log += time.time() - t0

                    if verbose: 
                        pbar.set_description(f"db_len={int(max_seq_length)} q_len={len(query)} matches={matches} max(SW)={max_sw} max(fft)={max_fft}")


        # list of lists -> list 
        t0 = time.time()
        seq_matchess = [[seq for batch in seq_matches for seq in batch] for seq_matches in seq_matchess]
        seq_scoress= [[score for batch in seq_scores for score in batch] for seq_scores in seq_scoress]
        seq_scores_ffts = [[score for batch in seq_scores_fft for score in batch] for seq_scores_fft in seq_scores_ffts]
        seq_scores_gaps = [[score for batch in seq_scores_gap for score in batch] for seq_scores_gap in seq_scores_gaps]

        for qi, (query, seq_matches, seq_scores, seq_scores_fft, seq_scores_gap) in enumerate(zip(queries, seq_matchess, seq_scoress, seq_scores_ffts, seq_scores_gaps)):

            if save_raw != '': 
                print(f"Saving to `{save_raw}`.")
                pickle.dump([query, seq_matches, seq_scores, seq_scores_fft, seq_scores_gap], open(save_raw, 'wb'))

            seq_matches, seq_scores = np.array(seq_matches), np.array(seq_scores)

            idx = np.argsort(seq_scores_gap)[::-1] 
            _seq_matches = seq_matches[idx][:num_msas]#//2] 

            if sync_time: th.cuda.synchronize()
            t_save = time.time()-t0

            t0 = time.time()

            if filenames[qi] is not None: 
                if verbose: 
                    print(f"Saving result to {filenames[qi]}")
                to_a3m(query, _seq_matches, filename=filenames[qi]) 

            t_a3m = time.time()-t0

        if return_timings:
            return {'t_dec': t_dec, 't_prep1': t_prep1, 't_prep2': t_prep2, 
                    't_fft': t_fft, 't_prep_fft': t_prep_fft, 't_sw': t_sw, 't_out': t_out, 't_out_cpu': t_out_cpu,
                    't_log': t_log, 't_topk1': t_topk1, 
                    't_topk2': t_topk2, 't_save': t_save,
                    't_prep_sw': t_prep_sw, 't_a3m': t_a3m, 't_to_vram': t_to_vram}

if __name__ == "__main__":
    msa('ADAM'*30, "adam.a3m")
    msa('ADA'*30, "ada.a3m")
