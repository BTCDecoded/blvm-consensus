#!/bin/bash
# Get list of slow Kani harnesses (unwind >= 10)
grep -r "kani::unwind" --include="*.rs" src/ | grep -E "unwind\((1[0-9]|20)\)" | while IFS=: read file line; do
  # Get function name (look 5 lines before unwind)
  func=$(sed -n "$(echo "$line" | cut -d: -f1)p" "$file" | grep -B 5 "unwind" | grep "fn kani_" | tail -1 | sed 's/.*fn \([a-zA-Z_]*\).*/\1/')
  # Get module name (look 20 lines before)
  mod=$(sed -n "1,$(echo "$line" | cut -d: -f1)p" "$file" | grep "mod kani_proofs" | tail -1 | sed 's/.*mod \([a-zA-Z_]*\).*/\1/')
  if [ -n "$func" ]; then
    if [ -n "$mod" ] && [ "$mod" != "kani_proofs" ]; then
      echo "${mod}::${func}"
    else
      echo "$func"
    fi
  fi
done | sort -u
