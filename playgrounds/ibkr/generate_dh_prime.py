import subprocess
import re

if __name__ == "__main__":
    result = subprocess.run(
        ["openssl", "dhparam", "-in", "dhparam.pem", "-text"],
        capture_output=True,
        text=True,
    ).stdout
    match = re.search(r"(?:prime|P):\s*((?:\s*[0-9a-fA-F:]+\s*)+)", result)
    print(
        re.sub(r"[\s:]", "", match.group(1)) if match else "No prime (P) value found."
    )
