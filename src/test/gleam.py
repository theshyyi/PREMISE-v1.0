import paramiko

host = "hydras.ugent.be"
port = 2225
username = "gleamuser"
password = "GLEAM4#h-cel_924"

transport = paramiko.Transport((host, port))
transport.connect(username=username, password=password)
sftp = paramiko.SFTPClient.from_transport(transport)

def safe_listdir(path: str):
    try:
        names = sftp.listdir(path)
        print(f"\n[path] {path}")
        for name in names[:50]:
            print("  ", name)
        if len(names) > 50:
            print(f"  ... ({len(names)} entries total)")
        return names
    except Exception as e:
        print(f"\n[path] {path}")
        print("  ERROR:", e)
        return []

# 先看你已经确认存在的目录
safe_listdir(".")
safe_listdir("./data")
safe_listdir("./v4")
safe_listdir("./v4/v4.2a")
safe_listdir("./v4/v4.2b")

# 继续往下试几个很可能的目录
candidates = [
    "./data/v4.2a",
    "./data/v4.2b",
    "./v4/v4.2a/daily",
    "./v4/v4.2b/daily",
    "./v4/v4.2a/monthly",
    "./v4/v4.2b/monthly",
    "./data/v4.2a/daily",
    "./data/v4.2b/daily",
    "./data/v4.2a/monthly",
    "./data/v4.2b/monthly",
]

for p in candidates:
    safe_listdir(p)

sftp.close()
transport.close()