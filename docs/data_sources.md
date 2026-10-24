# Data sources

All values below were reported from Shiyi's local runs on 2026-10-23.

## Narratives: archived official CFPB export

- Original URL: https://files.consumerfinance.gov/ccdb/complaints.csv.zip
- Archived copy: https://web.archive.org/web/20251209002532id_/https://files.consumerfinance.gov/ccdb/complaints.csv.zip
- Capture: Internet Archive, ArchiveTeam ArchiveBot, memento-datetime 2025-12-09 00:25:32 GMT
- Original Last-Modified: Sun, 07 Dec 2025 14:35:35 GMT
- Content-Length: 1,640,354,601 bytes, equal to the original server's Content-Length recorded in the capture
- SHA-256: 27350c1d6f297db7776851ebfa23caaedbc54dcc356e224d66e2cf5d41d2fba5
- Contents: one file, complaints.csv, 7,535,063,718 bytes, dated 2025-12-07; `unzip -t` reported no errors
- Includes the column `Consumer complaint narrative`
- Downloaded: 2026-10-23

## Labels and intake fields: current official CFPB export

- URL: https://files.consumerfinance.gov/ccdb/complaints.csv.zip
- Last-Modified: Fri, 23 Oct 2026 09:15:52 GMT
- Content-Length: 357,730,454 bytes; contents: complaints.csv, 5,644,741,497 bytes
- Downloaded: 2026-10-23
- Does not include `Consumer complaint narrative`. The CFPB search API answered
  `"complaint_what_happened" is not a valid choice` on 2026-10-23, so narratives are no
  longer available from official channels and cannot be re-downloaded from CFPB.
