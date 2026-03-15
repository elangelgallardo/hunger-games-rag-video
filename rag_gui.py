"""Tkinter GUI for the Hunger Games RAG Video Script Generator."""

import queue
import threading
import tkinter as tk
from tkinter import scrolledtext, ttk

from rag_query import generate_answer, get_ranker, hybrid_search, load_stores, rerank

BG_DARK = "#1a1a2e"
BG_LIGHT = "#f5f5f5"
ACCENT = "#e94560"
FONT_BODY = ("Helvetica", 11)
FONT_HEADER = ("Helvetica", 15, "bold")


class ScriptGeneratorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Hunger Games — Video Script Generator")
        self.geometry("960x720")
        self.minsize(640, 520)
        self.configure(bg=BG_DARK)

        self._collection = None
        self._ids = None
        self._docs = None
        self._bm25 = None
        self._ready = False
        self._q = queue.Queue()

        self._build_ui()
        self._load_stores_async()
        self.after(100, self._poll_queue)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        # ── Header ──────────────────────────────────────────────────────
        header = tk.Frame(self, bg=BG_DARK, pady=14)
        header.grid(row=0, column=0, sticky="ew")
        tk.Label(
            header,
            text="Hunger Games — Video Script Generator",
            font=FONT_HEADER,
            bg=BG_DARK,
            fg="white",
        ).pack()

        # ── Input panel ─────────────────────────────────────────────────
        input_frame = tk.Frame(self, bg=BG_DARK, padx=14, pady=6)
        input_frame.grid(row=1, column=0, sticky="ew")
        input_frame.columnconfigure(0, weight=1)

        tk.Label(input_frame, text="Topic", font=("Helvetica", 10), bg=BG_DARK, fg="#aaaaaa").grid(
            row=0, column=0, sticky="w"
        )

        entry_row = tk.Frame(input_frame, bg=BG_DARK)
        entry_row.grid(row=1, column=0, sticky="ew", pady=(2, 6))
        entry_row.columnconfigure(0, weight=1)

        self._topic_var = tk.StringVar()
        self._entry = ttk.Entry(entry_row, textvariable=self._topic_var, font=("Helvetica", 12))
        self._entry.grid(row=0, column=0, sticky="ew", ipady=4)
        self._entry.bind("<Return>", lambda _e: self._on_generate())

        btn_frame = tk.Frame(entry_row, bg=BG_DARK)
        btn_frame.grid(row=0, column=1, padx=(8, 0))

        self._gen_btn = tk.Button(
            btn_frame,
            text="Generate Script",
            command=self._on_generate,
            bg=ACCENT,
            fg="white",
            font=("Helvetica", 11, "bold"),
            relief="flat",
            padx=14,
            pady=4,
            cursor="hand2",
            state="disabled",
        )
        self._gen_btn.pack(side="left", padx=(0, 6))

        tk.Button(
            btn_frame,
            text="Clear",
            command=self._on_clear,
            bg="#444",
            fg="white",
            font=FONT_BODY,
            relief="flat",
            padx=10,
            pady=4,
            cursor="hand2",
        ).pack(side="left")

        self._debug_var = tk.BooleanVar()
        tk.Checkbutton(
            input_frame,
            text="Debug mode",
            variable=self._debug_var,
            bg=BG_DARK,
            fg="#aaaaaa",
            selectcolor=BG_DARK,
            activebackground=BG_DARK,
            font=("Helvetica", 10),
        ).grid(row=2, column=0, sticky="w")

        # ── Output area ─────────────────────────────────────────────────
        output_frame = tk.Frame(self, bg=BG_DARK, padx=14, pady=4)
        output_frame.grid(row=2, column=0, sticky="nsew")
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(1, weight=1)

        tk.Label(
            output_frame, text="Video Script", font=("Helvetica", 10), bg=BG_DARK, fg="#aaaaaa"
        ).grid(row=0, column=0, sticky="w")

        self._output = scrolledtext.ScrolledText(
            output_frame,
            font=("Georgia", 11),
            wrap=tk.WORD,
            bg="#1e2235",
            fg="#e8e8e8",
            insertbackground="white",
            relief="flat",
            padx=14,
            pady=14,
        )
        self._output.grid(row=1, column=0, sticky="nsew")

        tk.Button(
            output_frame,
            text="Copy to Clipboard",
            command=self._copy_to_clipboard,
            bg="#333",
            fg="#ccc",
            font=("Helvetica", 9),
            relief="flat",
            padx=8,
            pady=3,
            cursor="hand2",
        ).grid(row=2, column=0, sticky="e", pady=(6, 4))

        # ── Status bar ──────────────────────────────────────────────────
        status_bar = tk.Frame(self, bg="#111827", pady=5)
        status_bar.grid(row=3, column=0, sticky="ew")

        self._progress = ttk.Progressbar(status_bar, mode="indeterminate", length=120)
        self._progress.pack(side="right", padx=10)

        self._status_var = tk.StringVar(value="Loading stores…")
        tk.Label(
            status_bar,
            textvariable=self._status_var,
            bg="#111827",
            fg="#aaaaaa",
            anchor="w",
            font=("Helvetica", 9),
            padx=10,
        ).pack(side="left", fill="x", expand=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_status(self, msg: str):
        self._status_var.set(msg)

    def _write_output(self, text: str):
        self._output.delete("1.0", tk.END)
        self._output.insert(tk.END, text)

    def _copy_to_clipboard(self):
        text = self._output.get("1.0", tk.END).strip()
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)
            self._set_status("Copied to clipboard!")

    # ------------------------------------------------------------------
    # Background workers
    # ------------------------------------------------------------------

    def _load_stores_async(self):
        self._progress.start(10)

        def worker():
            try:
                collection, ids, docs, bm25 = load_stores()
                get_ranker()
                self._q.put(("stores_ready", (collection, ids, docs, bm25)))
            except Exception as exc:
                self._q.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_generate(self):
        if not self._ready:
            return
        topic = self._topic_var.get().strip()
        if not topic:
            self._set_status("Please enter a topic.")
            return

        self._gen_btn.config(state="disabled")
        self._progress.start(10)
        self._write_output("")
        debug = self._debug_var.get()

        def worker():
            try:
                self._q.put(("status", "Embedding query…"))
                candidates = hybrid_search(
                    topic, self._collection, self._ids, self._docs, self._bm25
                )

                self._q.put(("status", f"Reranking {len(candidates)} candidates…"))
                top_passages = rerank(topic, candidates)

                prefix = ""
                if debug:
                    lines = [f"{'─'*56}", "HYBRID SEARCH — top candidates", f"{'─'*56}"]
                    for i, c in enumerate(candidates[:10], 1):
                        preview = c["document"][:100].replace("\n", " ")
                        lines.append(f"{i}. [{c['id']}]  RRF={c['rrf']:.4f}  V={c['vector_score']:.3f}  B={c['bm25_score']:.3f}")
                        lines.append(f"   {preview}…")
                    lines += ["", f"{'─'*56}", "RERANKED", f"{'─'*56}"]
                    for i, p in enumerate(top_passages, 1):
                        preview = p["document"][:120].replace("\n", " ")
                        lines.append(f"{i}. [{p['id']}]  rerank={p['rerank_score']:.4f}")
                        lines.append(f"   {preview}…")
                    lines += ["", f"{'─'*56}", "CONTEXT SENT TO LLM", f"{'─'*56}"]
                    for i, p in enumerate(top_passages, 1):
                        lines.append(f"\n— Passage {i}: [{p['id']}] —")
                        lines.append(p["document"][:500])
                    lines += ["", f"{'─'*56}", "VIDEO SCRIPT", f"{'─'*56}", ""]
                    prefix = "\n".join(lines) + "\n"

                self._q.put(("status", "Generating script with LLM…"))
                answer = generate_answer(topic, top_passages)
                self._q.put(("result", prefix + answer))

            except Exception as exc:
                self._q.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_clear(self):
        self._topic_var.set("")
        self._write_output("")
        self._set_status("Ready.")
        self._entry.focus()

    # ------------------------------------------------------------------
    # Queue polling (runs on the main thread via after())
    # ------------------------------------------------------------------

    def _poll_queue(self):
        try:
            while True:
                msg_type, payload = self._q.get_nowait()
                if msg_type == "stores_ready":
                    self._collection, self._ids, self._docs, self._bm25 = payload
                    self._ready = True
                    self._gen_btn.config(state="normal")
                    self._progress.stop()
                    self._set_status(f"Ready — {len(self._ids):,} documents indexed.")
                    self._entry.focus()
                elif msg_type == "status":
                    self._set_status(payload)
                elif msg_type == "result":
                    self._progress.stop()
                    self._gen_btn.config(state="normal")
                    self._write_output(payload)
                    self._set_status("Done.")
                elif msg_type == "error":
                    self._progress.stop()
                    self._gen_btn.config(state="normal" if self._ready else "disabled")
                    self._set_status(f"Error: {payload}")
                    self._write_output(f"Error:\n\n{payload}")
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)


if __name__ == "__main__":
    app = ScriptGeneratorApp()
    app.mainloop()
