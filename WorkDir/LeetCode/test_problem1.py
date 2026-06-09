import subprocess

def run_test(s):
    print(f'Input: s = "{s}"')
    # 因为 text=True，input 应该是字符串而非 bytes
    result = subprocess.run(['python3', './WorkDir/LeetCode/problem1.py'], 
                              input=s, 
                              capture_output=True, 
                              text=True)
    print(f'Output: {result.stdout.strip()}')
    if result.stderr:
        print(f'Error: {result.stderr}')
    print('-' * 20)

if __name__ == "__main__":
    run_test("abcacbd")  # Expected: 1
    run_test("abc")       # Expected: 1
    run_test("abcdab")    # Expected: -1