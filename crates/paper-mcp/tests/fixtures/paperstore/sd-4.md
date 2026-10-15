---
title: "WG21 Practices and Procedures"
document: SD-4
date: 2024-09-02
audience: All of WG21
reply-to:
  - "Herb Sutter <herb@example.com>"
---

## Revision History

- 2024-09: Clarified the paper numbering rules.
- 2023-02: Initial publication.

## Paper Numbering

Papers are numbered Pnnnn with a revision suffix Rn. Standing documents use the SD-n form and are revised in place.

| Prefix | Kind | Revised |
| --- | --- | --- |
| P | proposal | by new revision |
| N | official document | never |
| SD | standing document | in place |

```cpp
// Feature-test macros are named __cpp_lib_<feature>.
#define __cpp_lib_ring_span 202501L
```

## Wording

Proposals targeting the working draft must include a Wording section with changes relative to the most recent N-numbered working draft.
