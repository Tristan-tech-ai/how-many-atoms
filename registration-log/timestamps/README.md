# Time stamps of the registration log

Each `stamp_<date>T<time>.txt` is a manifest. It gives the byte length and the SHA-256 hash of the original registration log and of its public
copy at the time it was written, and the number of the last entry. Only the hash of a manifest was sent out, to two independent services:

- OpenTimestamps: `stamp_*.txt.ots` is a proof that the manifest existed when a Bitcoin block was mined. Check it with the OpenTimestamps
  client (`pip install opentimestamps-client`): `ots verify stamp_<date>T<time>.txt.ots`. A proof that is still pending reports so; `ots upgrade`
  completes it once the block exists. Without a Bitcoin node, `ots --no-bitcoin verify` prints the block and the merkle root that the
  proof requires, and any block explorer shows the merkle root of that block. Each `verify_*.txt` records one such check: the block, the merkle
  root that the proof requires, and the merkle root that two block explorers give for that block.
- RFC 3161 (FreeTSA): `stamp_*.txt.tsr` is a signed time-stamp token for the request `stamp_*.txt.tsq`. Check it with OpenSSL:
  `openssl ts -verify -in stamp_<date>T<time>.txt.tsr -queryfile stamp_<date>T<time>.txt.tsq -CAfile cacert.pem -untrusted tsa.crt`.
  The certificates `cacert.pem` and `tsa.crt` are those published at https://freetsa.org/files/.

The original log is append-only. A later copy of it can be checked against a manifest by hashing its first N bytes, with N from the manifest. The
public copy is checked against the copy of the same date in the history of this repository. Entries written before the first manifest carry
only the dates of the log itself.
