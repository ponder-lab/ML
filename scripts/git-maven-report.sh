#!/bin/bash

COMMIT_RANGE="45355076e1ac7b1924059cb596e2868bcef81710~..HEAD"

# Output CSV file name
OUTPUT_FILE="maven_build_stats.csv"

# Initialize CSV header
echo "Commit Hash,Author,Date,Message,Failures,Errors" > "$OUTPUT_FILE"

# Get list of commits (reverse chronological order)
# You can limit the number of commits by adding -n <number> to git log
COMMITS=$(git log $COMMIT_RANGE --pretty=format:"%H" --reverse)

# Save current branch name to restore later
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

echo "Starting analysis... Output will be saved to $OUTPUT_FILE"

# Loop through each commit
for COMMIT in $COMMITS; do
    echo "Processing commit: $COMMIT"

    # Checkout the commit
    git checkout "$COMMIT" 2>/dev/null

    if [ $? -ne 0 ]; then
        echo "Error checking out commit $COMMIT. Skipping."
        continue
    fi

    # Get commit metadata
    AUTHOR=$(git show -s --format="%an" "$COMMIT")
    DATE=$(git show -s --format="%ad" --date=short "$COMMIT")

    # Get the raw subject line (first line of message)
    RAW_MESSAGE=$(git show -s --format="%s" "$COMMIT")

    # Sanitize message for CSV:
    # 1. Escape double quotes (replace " with "")
    SAFE_MESSAGE=$(echo "$RAW_MESSAGE" | sed 's/"/""/g')

    # Run Maven build (clean verify usually covers tests)
    # capturing output to a variable. We use -fn (fail-never) so the script
    # doesn't exit if the build fails.
    MVN_OUTPUT=$(mvn clean test -fn -B 2>&1)

    # Extract failures and errors using grep and awk
    # Looking for pattern: "Tests run: X, Failures: Y, Errors: Z, Skipped: W"
    # We sum them up because a multi-module project outputs this line multiple times.

    # --- PARSING FIX ---
    # 1. grep "Tests run:" -> Find all lines with test counts
    # 2. tail -n 1       -> Keep ONLY the very last line found
    # 3. sed             -> Extract the number

    LAST_LINE=$(echo "$MVN_OUTPUT" | grep "Tests run:" | tail -n 1)

    # Extract Failures from that single line (default to 0 if empty)
    FAILURES=$(echo "$LAST_LINE" | sed -n 's/.*Failures: \([0-9]\+\).*/\1/p')
    FAILURES=${FAILURES:-0}

    # Extract Errors from that single line (default to 0 if empty)
    ERRORS=$(echo "$LAST_LINE" | sed -n 's/.*Errors: \([0-9]\+\).*/\1/p')
    ERRORS=${ERRORS:-0}

    # Append to CSV
    echo "$COMMIT,\"$AUTHOR\",$DATE,\"$SAFE_MESSAGE\",$FAILURES,$ERRORS" >> "$OUTPUT_FILE"

done

# Restore original branch
echo "Analysis complete. Restoring branch $CURRENT_BRANCH..."
git checkout "$CURRENT_BRANCH" 2>/dev/null

echo "Done. Results saved in $OUTPUT_FILE"
