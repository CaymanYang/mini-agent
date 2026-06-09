def solve():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    s = input_data[0]
    n = len(s)
    
    for i in range(n):
        if s[i] == s[n - i - 1]:
            print(i)
            return
            
    print(-1)

if __name__ == "__main__":
    solve()