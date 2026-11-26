import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter.scrolledtext import ScrolledText

BASE_DIR = Path(__file__).resolve().parent
PYTHON_BIN = BASE_DIR / ".venv" / "Scripts" / "python.exe"
MODEL_PATH = BASE_DIR / "checkpoint_v4.pt"
SAMPLE_SCRIPT = BASE_DIR / "sample.py"


def send_message(event=None):
    prompt = entry.get().strip()
    if not prompt or send_btn["state"] == "disabled":
        return

    entry.delete(0, tk.END)
    chat.insert(tk.END, f"\nTy: {prompt}\n...\n")
    chat.see(tk.END)
    send_btn.config(state="disabled")

    def generate():
        cmd = [
            str(PYTHON_BIN),
            str(SAMPLE_SCRIPT),
            "--checkpoint", str(MODEL_PATH),
            "--prompt", prompt,
            "--tokens", "100",
            "--temperature", "0.7",
            "--top-k", "40",
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(BASE_DIR),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=180,
            )
            answer = result.stdout.strip() if result.returncode == 0 else (result.stderr.strip() or "Error.")
        except Exception as e:
            answer = f"Error: {e}"

        def update_ui():
            chat.insert(tk.END, f"\n{answer}\n")
            chat.see(tk.END)
            send_btn.config(state="normal")
            entry.focus()

        # Tkinter requires UI updates on the main thread
        root.after(0, update_ui)

    threading.Thread(target=generate, daemon=True).start()


# --- UI Setup ---
root = tk.Tk()
root.title("Tiny DK Chat")
root.geometry("650x500")
root.configure(bg="#202532")

chat = ScrolledText(
    root,
    wrap=tk.WORD,
    font=("Consolas", 11),
    bg="#151923",
    fg="white",
    insertbackground="white",
)
chat.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

bottom = tk.Frame(root, bg="#202532")
bottom.pack(fill=tk.X, padx=10, pady=(0, 10))

entry = tk.Entry(
    bottom,
    font=("Segoe UI", 12),
    bg="#303747",
    fg="white",
    insertbackground="white",
)
entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 8))
entry.bind("<Return>", send_message)

send_btn = tk.Button(
    bottom,
    text="Send",
    command=send_message,
    bg="#4268f5",
    fg="white",
    font=("Segoe UI", 11),
)
send_btn.pack(side=tk.RIGHT, ipady=5)

entry.focus()
root.mainloop()
