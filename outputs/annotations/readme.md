We distinguish between the following types of metadata presence (on the dataset landing page):
- 0: Not available anywhere.
- 1: Available and clearly structured. (E.g., "License: CC-BY-NC" clearly indicates that "CC-BY-NC" is the _license_.)
- 2: Available but not clearly structured. (E.g., If the description says "data was collected bi-monthly from ..", this does not clearly indicate that "bi-monthly" is the _temporal resolution_).

Annotations should follow the following (yaml) format:

URL 1
  Metadata field 1:
    text: Metadata
    findability: 1
  Metadata field 2:
    Metadata
    findability: 1
URL 2
  Metadata field 1:
....


