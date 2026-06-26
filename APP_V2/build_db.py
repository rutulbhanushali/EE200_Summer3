"""build_db.py  –  One-time script to index the songs/ folder into db.pkl
EE200: Signals, Systems & Networks – Course Project (Q3B)

Run this once before launching the Streamlit app (or just let the app
build the DB on first startup if you prefer).

Usage
─────
  python build_db.py [--songs <folder>] [--out <db_path>]

Defaults: songs folder = "songs/", output = "db.pkl"
"""

import argparse
import time
import fingerprint as fp

def main():
    parser = argparse.ArgumentParser(description="Build audio fingerprint database.")
    parser.add_argument("--songs", default="songs",
                        help="Folder containing song audio files (default: songs/)")
    parser.add_argument("--out", default="db.pkl",
                        help="Output path for the database pickle (default: db.pkl)")
    args = parser.parse_args()

    print(f"Building fingerprint database from '{args.songs}' …")
    t0 = time.time()

    db = fp.FingerprintDB()
    n  = db.build_from_folder(args.songs)

    if n == 0:
        print(f"  WARNING: No audio files found in '{args.songs}/'. "
              "Make sure the songs are in that directory.")
        return

    db.save(args.out)
    elapsed = time.time() - t0
    total_h = sum(db.songs.values())

    print(f"  Indexed {n} song(s)  ·  {total_h:,} total hashes  "
          f"·  saved to '{args.out}'  ({elapsed:.1f}s)")
    print()
    print("Song breakdown:")
    for label, count in sorted(db.songs.items()):
        print(f"  {label:<40s}  {count:>8,} hashes")

if __name__ == "__main__":
    main()
