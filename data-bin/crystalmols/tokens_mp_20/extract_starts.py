"""Extract per-CIF start offsets from a tokenized ``*.bin`` file.

The ``starts_<dataset>_<split>.pkl`` files consumed by
``byprot.datamodules.crystalmols_hf.CrystalDataset`` are produced by this
utility. Each CIF in the bin stream begins at a ``data_`` token, so the offsets
of those tokens are exactly the row boundaries.

Run it inside a tokenized dataset directory, e.g.::

    cd data-bin/crystalmols/tokens_mp_20
    python extract_starts.py

It is idempotent: re-running it rewrites the ``starts_*.pkl`` files with the
same offsets, and never modifies the ``.bin`` data.
"""

import pickle

import numpy as np


def extract_cif_starts_from_bin(bin_path, meta_path, output_path=None, verbose=True):
    """Extract the start offset of every CIF in a bin file (the ``data_`` token positions).

    The bin format is a plain ``np.uint16`` array serialized to binary.

    Args:
        bin_path: path to the .bin file (e.g. 'val.bin' or 'train.bin')
        meta_path: path to meta.pkl (holds the token-to-id mapping)
        output_path: path of the output .pkl file (e.g. 'starts_mp_20_val.pkl')
        verbose: whether to print details

    Returns:
        list: the start offset of each CIF
    """
    # 1. Read the bin file (uint16 array)
    bin_data = np.fromfile(bin_path, dtype=np.uint16)

    if verbose:
        print(f"read {bin_path}: {len(bin_data)} tokens")
        print(f"dtype: {bin_data.dtype}")
        print(f"first 10 tokens (ids): {bin_data[:10].tolist()}")

    # 2. Read meta.pkl to get the token id of 'data_'
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    # Parse the meta format (several storage layouts are supported)
    data_token_id = None
    if isinstance(meta, dict):
        # try the known key names
        if 'vocab' in meta:
            vocab = meta['vocab']
            if isinstance(vocab, dict):
                data_token_id = vocab.get('data_', None)
        elif 'stoi' in meta:
            data_token_id = meta['stoi'].get('data_', None)
        elif 'token_to_id' in meta:
            data_token_id = meta['token_to_id'].get('data_', None)
        else:
            # look the token up directly in meta
            data_token_id = meta.get('data_', None)

    if data_token_id is None:
        raise ValueError("could not find the token id of 'data_' in meta.pkl; check the meta format")

    if verbose:
        print(f"'data_' token id: {data_token_id}")

    # 3. Find every 'data_' token position
    starts = []
    for i, token in enumerate(bin_data):
        if token == data_token_id:
            starts.append(i)

    if verbose:
        print(f"found {len(starts)} CIF records")

    # 4. Sanity check: the first record must start at offset 0
    if len(starts) > 0:
        if starts[0] == 0:
            if verbose:
                print("OK: the first CIF starts at offset 0")
        else:
            print(f"WARNING: the first CIF starts at offset {starts[0]}, not 0")

    # 5. Save the start offsets
    if output_path:
        with open(output_path, 'wb') as f:
            pickle.dump(starts, f)
        if verbose:
            print(f"saved start offsets to: {output_path}")

    return starts


def inspect_bin_file(bin_path, meta_path, num_samples=20):
    """Print a summary of a .bin file (read with the correct uint16 dtype).

    Args:
        bin_path: path to the .bin file
        meta_path: path to meta.pkl
        num_samples: how many entries to display
    """

    print("=" * 70)
    print(f"inspecting file: {bin_path}")
    print("=" * 70)

    # 1. Read the bin file (uint16)
    bin_data = np.fromfile(bin_path, dtype=np.uint16)

    print(f"\nfile size: {len(bin_data):,} tokens")
    print(f"dtype: {bin_data.dtype}")
    print(f"value range: [{bin_data.min()}, {bin_data.max()}]")

    # 2. Show the first tokens
    print(f"\nfirst {num_samples} tokens (ids):")
    for i in range(min(num_samples, len(bin_data))):
        print(f"  [{i:3d}] {bin_data[i]:6d}")

    # 3. Read meta.pkl and decode
    print(f"\n" + "=" * 70)
    print("decoding tokens with meta.pkl")
    print("=" * 70)

    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    # Build the id_to_token mapping
    id_to_token = {}
    if isinstance(meta, dict):
        if 'vocab' in meta and isinstance(meta['vocab'], dict):
            id_to_token = {v: k for k, v in meta['vocab'].items()}
            print(f"loaded {len(id_to_token)} tokens from 'vocab'")
        elif 'itos' in meta:
            id_to_token = {i: t for i, t in enumerate(meta['itos'])}
            print(f"loaded {len(id_to_token)} tokens from 'itos'")
        elif 'stoi' in meta:
            id_to_token = {v: k for k, v in meta['stoi'].items()}
            print(f"loaded {len(id_to_token)} tokens from 'stoi'")
        else:
            id_to_token = {v: k for k, v in meta.items()}
            print(f"using meta directly: {len(id_to_token)} tokens")

    # Show the decoded form of the first tokens
    print(f"\nfirst {num_samples} tokens (decoded):")
    for i in range(min(num_samples, len(bin_data))):
        token_id = bin_data[i]
        token_str = id_to_token.get(int(token_id), f"<UNKNOWN:{token_id}>")
        print(f"  [{i:3d}] {token_id:6d} -> '{token_str}'")

    # 4. Locate the 'data_' token
    data_token_id = None
    if isinstance(meta, dict):
        if 'vocab' in meta and isinstance(meta['vocab'], dict):
            for k, v in meta['vocab'].items():
                if k == 'data_':
                    data_token_id = v
                    break
        elif 'stoi' in meta:
            data_token_id = meta['stoi'].get('data_')

    if data_token_id is not None:
        print(f"\n'data_' token id: {data_token_id}")

        # Find every 'data_' position
        data_positions = np.where(bin_data == data_token_id)[0]
        print(f"'data_' occurrences: {len(data_positions)}")

        if len(data_positions) > 0:
            print(f"first 20 occurrences: {data_positions[:20].tolist()}")

            if data_positions[0] == 0:
                print("OK: the first token is 'data_'")
            else:
                print(f"WARNING: the first 'data_' is at offset {data_positions[0]}, not 0")

    return bin_data


def verify_starts(bin_path, starts_path, meta_path, num_samples=20):
    """Verify that the extracted start offsets are correct.

    Args:
        bin_path: path to the .bin file
        starts_path: path to the .pkl file holding the start offsets
        meta_path: path to meta.pkl
        num_samples: how many samples to verify
    """
    # 1. Read the data (uint16)
    bin_data = np.fromfile(bin_path, dtype=np.uint16)

    with open(starts_path, 'rb') as f:
        starts = pickle.load(f)

    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    # Build the id_to_token mapping
    id_to_token = {}
    if isinstance(meta, dict):
        if 'vocab' in meta and isinstance(meta['vocab'], dict):
            id_to_token = {v: k for k, v in meta['vocab'].items()}
        elif 'itos' in meta:
            id_to_token = {i: t for i, t in enumerate(meta['itos'])}
        elif 'stoi' in meta:
            id_to_token = {v: k for k, v in meta['stoi'].items()}
        else:
            id_to_token = {v: k for k, v in meta.items()}

    # Get the 'data_' token id
    data_token_id = None
    if isinstance(meta, dict):
        if 'vocab' in meta and isinstance(meta['vocab'], dict):
            for k, v in meta['vocab'].items():
                if k == 'data_':
                    data_token_id = v
                    break
        elif 'stoi' in meta:
            data_token_id = meta['stoi'].get('data_')

    if data_token_id is None:
        print("cannot verify: 'data_' token id not found")
        return False

    print(f"\n{'='*70}")
    print(f"verifying file: {bin_path}")
    print(f"total CIF records: {len(starts)}")
    print(f"verifying the first {num_samples} start offsets")
    print(f"{'='*70}\n")

    all_valid = True
    for i in range(min(num_samples, len(starts))):
        pos = starts[i]

        if pos >= len(bin_data):
            print(f"FAIL CIF {i}: offset {pos} is out of range")
            all_valid = False
            continue

        token_id = bin_data[pos]
        token_str = id_to_token.get(int(token_id), f"<UNKNOWN:{token_id}>")
        is_data = (token_id == data_token_id)
        status = "OK  " if is_data else "FAIL"

        print(f"{status} CIF {i:3d}: offset {pos:8d} -> token id {token_id:5d} -> '{token_str}'")

        if not is_data:
            all_valid = False

    if all_valid:
        print("\nall checks passed")
    else:
        print("\nverification FAILED")

    return all_valid


# ==================== main ====================
if __name__ == "__main__":
    # Paths are relative to the dataset directory this script lives in.
    data_dir = "."

    train_bin = f"{data_dir}/train.bin"
    val_bin = f"{data_dir}/val.bin"
    meta_file = f"{data_dir}/meta.pkl"

    train_starts_output = "starts_mp_20_train.pkl"
    val_starts_output = "starts_mp_20_val.pkl"

    # 1. Inspect train.bin
    print("\n" + "=" * 70)
    print("inspecting train data")
    print("=" * 70)
    train_data = inspect_bin_file(train_bin, meta_file, num_samples=20)

    # 2. Inspect val.bin
    print("\n" + "=" * 70)
    print("inspecting val data")
    print("=" * 70)
    val_data = inspect_bin_file(val_bin, meta_file, num_samples=20)

    # 3. Extract train start offsets
    print("\n" + "=" * 70)
    print("extracting train CIF start offsets")
    print("=" * 70)
    train_starts = extract_cif_starts_from_bin(
        train_bin,
        meta_file,
        train_starts_output,
        verbose=True
    )

    # 4. Extract val start offsets
    print("\n" + "=" * 70)
    print("extracting val CIF start offsets")
    print("=" * 70)
    val_starts = extract_cif_starts_from_bin(
        val_bin,
        meta_file,
        val_starts_output,
        verbose=True
    )

    # 5. Verify train starts
    print("\n" + "=" * 70)
    print("verifying train starts")
    print("=" * 70)
    verify_starts(train_bin, train_starts_output, meta_file, num_samples=20)

    # 6. Verify val starts
    print("\n" + "=" * 70)
    print("verifying val starts")
    print("=" * 70)
    verify_starts(val_bin, val_starts_output, meta_file, num_samples=20)

    # 7. Print the first 20 offsets
    print("\n" + "=" * 70)
    print("first 20 train starts:")
    print("=" * 70)
    for i, pos in enumerate(train_starts[:20]):
        print(f"  [{i:3d}] {pos:8d}")

    print("\n" + "=" * 70)
    print("first 20 val starts:")
    print("=" * 70)
    for i, pos in enumerate(val_starts[:20]):
        print(f"  [{i:3d}] {pos:8d}")

    print("\n" + "=" * 70)
    print("done")
    print("=" * 70)
