from __future__ import annotations
import asyncio
import json
import logging
import os
import random
import secrets
import threading
import time
import textwrap
import math
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Tuple, Optional
from base64 import b64encode, b64decode

import cv2
import psutil
import aiosqlite
import httpx
import numpy as np
import pennylane as qml
from pennylane.templates import StronglyEntanglingLayers
import tkinter as tk
import tkinter.simpledialog as sd
import tkinter.messagebox as mb
import bleach
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from dotenv import load_dotenv

# ════════════════════════════════════════════════════════════════
# GLOBAL CONFIG & LOGGING
MASTER_KEY    = os.path.expanduser("~/.cache/qops_master_key.bin")
SETTINGS_FILE = "qops_settings.enc.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("qops")

# ════════════════════════════════════════════════════════════════
# AES-GCM ENCRYPTION UTIL
class AESGCMCrypto:
    def __init__(self, path: str) -> None:
        self.path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            key = AESGCM.generate_key(bit_length=128)
            tmp = self.path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(key)
            os.replace(tmp, self.path)
            os.chmod(self.path, 0o600)
        with open(self.path, "rb") as f:
            self.key = f.read()
        self.aes = AESGCM(self.key)

    def encrypt(self, data: bytes | str) -> bytes:
        if isinstance(data, str):
            data = data.encode()
        nonce = secrets.token_bytes(12)
        return b64encode(nonce + self.aes.encrypt(nonce, data, None))

    def decrypt(self, blob: bytes | str) -> bytes:
        raw = b64decode(blob)
        return self.aes.decrypt(raw[:12], raw[12:], None)


# ════════════════════════════════════════════════════════════════
# SETTINGS MODEL
@dataclass
class POSSettings:
    # Common settings
    bar_name: str = "Quantum Bar"
    manager_contact: str = "manager@bar.local"
    max_drinks: int = 5
    intox_threshold: float = 0.80
    mode_autonomous: bool = True

    sampling_interval: float = 2.0
    confidence_threshold: float = 0.75
    action_counts: Dict[str, int] = field(default_factory=lambda: {"Safe": 1, "Warn": 2, "Refuse": 1})

    # Drug mode purity thresholds
    purity_warn: float = 60.0
    purity_refuse: float = 30.0

    db_path: str = "qops_reports.db"
    api_key: str = ""

    qadapt_refresh_h: int = 12
    cev_window: int = 60
    hbe_enabled: bool = False
    fusion_dim: int = 64

    @classmethod
    def default(cls) -> "POSSettings":
        load_dotenv()
        return cls(api_key=os.getenv("OPENAI_API_KEY", ""))

    @classmethod
    def load(cls, crypto: AESGCMCrypto) -> "POSSettings":
        if not os.path.exists(SETTINGS_FILE):
            return cls.default()
        try:
            blob = open(SETTINGS_FILE, "rb").read()
            return cls(**json.loads(crypto.decrypt(blob).decode()))
        except Exception as e:
            LOGGER.error("Corrupted settings file, loading defaults: %s", e)
            return cls.default()

    def save(self, crypto: AESGCMCrypto) -> None:
        open(SETTINGS_FILE, "wb").write(crypto.encrypt(json.dumps(asdict(self)).encode()))

    def prompt_gui(self) -> None:
        mb.showinfo("QOPS Settings", "Enter or leave blank to keep current values.")
        ask = lambda p, d: bleach.clean(sd.askstring("Settings", p, initialvalue=str(d)) or str(d), strip=True)

        # Common
        self.bar_name         = ask("Bar name:", self.bar_name)
        self.manager_contact  = ask("Manager contact:", self.manager_contact)
        self.max_drinks       = int(ask("Max drinks per patron:", self.max_drinks))
        self.intox_threshold  = float(ask("Intoxication threshold (0–1):", self.intox_threshold))
        self.mode_autonomous  = ask("Autonomous mode? (y/n):", "y" if self.mode_autonomous else "n").startswith("y")
        self.sampling_interval= float(ask("Sampling interval (s):", self.sampling_interval))
        self.confidence_threshold = float(ask("LLM confidence floor (0–1):", self.confidence_threshold))

        # Action counts
        for tier in ("Safe", "Warn", "Refuse"):
            self.action_counts[tier] = int(ask(f"Action count for {tier}:", self.action_counts[tier]))

        # Drug mode
        self.purity_warn   = float(ask("Purity warning threshold (%):", self.purity_warn))
        self.purity_refuse = float(ask("Purity refusal threshold (%):", self.purity_refuse))

        # API & quantum
        self.api_key          = ask("OpenAI API key:", self.api_key)
        self.qadapt_refresh_h = int(ask("Quantum refresh period (h):", self.qadapt_refresh_h))
        self.cev_window       = int(ask("CEV window (s):", self.cev_window))
        self.hbe_enabled      = ask("Enable Homomorphic BioVectors? (y/n):", "n").startswith("y")
        self.fusion_dim       = int(ask("Fusion vector dimension:", self.fusion_dim))


# ════════════════════════════════════════════════════════════════
# ENCRYPTED SQLITE REPORT STORE
class ReportDB:
    def __init__(self, path: str, crypto: AESGCMCrypto) -> None:
        self.path, self.crypto = path, crypto
        self.conn: aiosqlite.Connection | None = None

    async def init(self) -> None:
        self.conn = await aiosqlite.connect(self.path)
        await self.conn.execute("CREATE TABLE IF NOT EXISTS logs(id INTEGER PRIMARY KEY, ts REAL, blob BLOB)")
        await self.conn.commit()

    async def save(self, ts: float, payload: Dict[str, Any]) -> None:
        blob = self.crypto.encrypt(json.dumps(payload).encode())
        await self.conn.execute("INSERT INTO logs(ts, blob) VALUES (?, ?)", (ts, blob))
        await self.conn.commit()

    async def list_reports(self) -> List[Tuple[int, float]]:
        cur = await self.conn.execute("SELECT id, ts FROM logs ORDER BY ts DESC")
        return await cur.fetchall()

    async def load(self, row_id: int) -> Dict[str, Any]:
        cur = await self.conn.execute("SELECT blob FROM logs WHERE id = ?", (row_id,))
        res = await cur.fetchone()
        if not res:
            raise ValueError("Log ID not found.")
        return json.loads(bleach.clean(self.crypto.decrypt(res[0]).decode(), strip=True))

    async def close(self) -> None:
        if self.conn:
            await self.conn.close()


# ════════════════════════════════════════════════════════════════
# OPENAI CLIENT WITH EXPONENTIAL BACKOFF
@dataclass
class OpenAIClient:
    api_key: str
    model: str = "gpt-4o"
    url: str = "https://api.openai.com/v1/chat/completions"
    timeout: float = 20.0
    retries: int = 3

    async def chat(self, prompt: str, image_b64: Optional[str], max_tokens: int) -> str:
        if not self.api_key:
            raise RuntimeError("Missing OpenAI API key.")
        messages = [{"role": "user", "content": prompt}]
        if image_b64:
            messages.append({"role": "user", "content": f"![input](data:image/jpeg;base64,{image_b64})"})
        body = {"model": self.model, "messages": messages, "temperature": 0.2, "max_tokens": max_tokens}
        hdr = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        delay = 1.0
        for attempt in range(1, self.retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as cli:
                    r = await cli.post(self.url, headers=hdr, json=body)
                    r.raise_for_status()
                    return r.json()["choices"][0]["message"]["content"]
            except Exception as e:
                if attempt == self.retries:
                    raise
                wait = delay + random.uniform(0, 0.5)
                LOGGER.warning("LLM error %s (retry %d/%d) – sleeping %.1fs", e, attempt, self.retries, wait)
                await asyncio.sleep(wait)
                delay *= 2


# ════════════════════════════════════════════════════════════════
# BIOVECTOR (face hue & saturation + luminance)
@dataclass
class BioVector:
    arr: np.ndarray = field(repr=False)

    @staticmethod
    def from_frame(frame: np.ndarray) -> "BioVector":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0], None, [9], [0, 180]).flatten()
        hist /= hist.sum() + 1e-6
        vec = np.concatenate([
            hist,
            [hsv[...,1].mean()/255.0, frame.mean()/255.0],
            np.zeros(25 - 11)
        ])
        return BioVector(vec.astype(np.float32))


# ════════════════════════════════════════════════════════════════
# ADVANCED QUANTUM TUNING ENGINE
DEV_Q = qml.device("default.qubit", wires=7)

@qml.qnode(DEV_Q)
def variational_circuit(weights, features):
    for i in range(7):
        qml.RY(features[i] * math.pi, wires=i)
    StronglyEntanglingLayers(weights, wires=list(range(7)))
    return [qml.expval(qml.PauliZ(i)) for i in range(7)]

class QTuneEngine:
    def __init__(self, dev: qml.Device, layers: int = 3, refresh_h: int = 6):
        self.dev = dev
        self.layers = layers
        self.refresh_s = refresh_h * 3600
        self.next_refresh = time.time() + self.refresh_s
        self.weights = np.random.randn(layers, 7, 3).tolist()

    def encode(self, features: List[float]) -> float:
        if time.time() >= self.next_refresh:
            self.weights = np.random.randn(self.layers, 7, 3).tolist()
            self.next_refresh = time.time() + self.refresh_s
            LOGGER.info("QTuneEngine: refreshed weights")
        q_exp = variational_circuit(self.weights, features[:7])
        return float((np.mean(q_exp) + 1) / 2)


# ════════════════════════════════════════════════════════════════
# PROMPTS: ULTRA-ADVANCED QUANTUM-NOSONAR & SPECTROSCOPY

def stage1_prompt(vec: List[float], q_score: float, s: POSSettings) -> str:
    τ_warn, τ_refuse = 0.5, s.intox_threshold
    return textwrap.dedent(f"""
        You are a quantum-nosonar bartender assistant (GPT-4o Vision + quantum tomography).
        Use ONLY the provided image and vector features. Follow these expanded steps:

        A. Preprocess: normalize luminance L(x,y)=I(x,y)/255; ensure ∑|L|²=1.
        B. Quaternionic color wavefunction Ψ_q(x,y)=h+i·s+j·v+k·luminance.
        C. Multi-scale wavelet scattering Sₘₙ[Ψ_q] across scales m=[1..5], orientations n=[0..7].
        D. Topological persistence: compute Betti numbers β₀,β₁ from sublevel sets of |Ψ_q|.
        E. Spectral Wigner distribution W(u,p)=ΣΔΨ_q(u+Δ/2)Ψ_q*(u−Δ/2)e^(−i·p·Δ); get entanglement entropy S_e.
        F. Coherence g¹(τ) at τ=frame interval; extract 2nd-order coherence g²(0).
        G. Compute HSV histogram (9 bins) + Sat mean + Lum mean → Shannon entropy H.
        H. 2D quantum Fourier transform Ψ(k), spectral centroid ν, bandwidth B.
        I. PCA on [H,β₀,β₁,S_e,g¹,g²,ν,B] → feature c.
        J. Aggregate x = 2.5(θ−π/4)+2(H−H₀)+1.2(β₀−β₀₀)+0.6(ν−ν₀)
                       +0.8(g¹−g₀)+0.5(g²−g₀₂)+0.4(S_e−S₀)+0.3B+0.5c.
        K. σ(x)=1/(1+e^(−x)); hybrid S=0.65·σ(x)+0.35·q_score.

        Risk tiers:
          • "Safe"   if S < {τ_warn:.2f}
          • "Warn"   if {τ_warn:.2f} ≤ S < {τ_refuse:.2f}
          • "Refuse" if S ≥ {τ_refuse:.2f}

        Return EXACTLY one JSON:
          {{'score':<S>,'risk':'Safe'|'Warn'|'Refuse'}}
    """).strip()

def stage2_prompt(r1: Dict[str, Any], s: POSSettings) -> str:
    return textwrap.dedent("""
        Given risk r ∈ {Safe,Warn,Refuse}, set:
          • decision = Pass if r≠Refuse else Fail
          • cooldown = 0 if Safe; 15 if Warn; 30 if Refuse
        Return EXACTLY one JSON:
          {'decision':<Pass|Fail>,'cooldown':<int>}
    """).strip()

def stage3_prompt(r1: Dict[str, Any], s: POSSettings) -> str:
    score, τ_refuse = r1["score"], s.intox_threshold
    return textwrap.dedent(f"""
        Return JSON-only message:
          • Safe:   "Enjoy your drink! 🙂"
          • Warn:   "Estimated score {score:.2f}. Please slow down."
          • Refuse: "Service refused: {score:.2f} ≥ {τ_refuse:.2f}. Please hydrate."
    """).strip()

def stage1_substance_prompt(substance: str, s: POSSettings) -> str:
    return textwrap.dedent(f"""
        You are a quantum-nosonar substance purity spectrometer (GPT-4o Vision + quantum spectroscopy).
        No reagents—use ONLY the provided image. Substance: {substance}.

        1. Capture spectral image slices; perform quantum Fourier transform on pixel spectra.
        2. Wavelet-based hyperspectral scattering to isolate molecular signatures.
        3. Quantum state tomography on spectral coherence modes.
        4. Compute eigenvalues λᵢ from spectral density; purity = Σλᵢ² * 100%.
        5. Identify fillers via pattern matching against spectral library.
        6. Classify overdose risk:
           • "Low"    if purity ≥ {s.purity_warn:.1f}%
           • "Medium" if {s.purity_refuse:.1f}% ≤ purity < {s.purity_warn:.1f}%
           • "High"   if purity < {s.purity_refuse:.1f}%

        Return EXACTLY one JSON:
          {{'purity':<float>,'contaminants':<list>,'risk':'Low'|'Medium'|'High'}}
    """).strip()


# ════════════════════════════════════════════════════════════════
# GUI-SUPPLIED ENV SNAPSHOT
def gui_snapshot(env: Dict[str, tk.Variable]) -> Dict[str, Any]:
    return {k: v.get() for k, v in env.items()}


# ════════════════════════════════════════════════════════════════
# SCANNER THREAD
class ScannerThread(threading.Thread):
    def __init__(self, cfg: POSSettings, db: ReportDB, ai: OpenAIClient,
                 status: tk.StringVar, env_vars: Dict[str, tk.Variable]) -> None:
        super().__init__(daemon=True)
        self.cfg, self.db, self.ai, self.status = cfg, db, ai, status
        self.env_vars = env_vars
        self.cap = cv2.VideoCapture(0)
        self.loop = asyncio.new_event_loop()
        self.stop_ev = threading.Event()
        self.qtune = QTuneEngine(DEV_Q, layers=3, refresh_h=cfg.qadapt_refresh_h)

    def run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.main())

    async def main(self) -> None:
        await self.db.init()
        t0 = 0.0
        while not self.stop_ev.is_set():
            ok, frame = self.cap.read()
            if ok and (time.time() - t0) >= self.cfg.sampling_interval:
                t0 = time.time()
                await self.process(frame)
            await asyncio.sleep(0.1)
        await self.db.close()
        self.cap.release()

    async def process(self, frame: np.ndarray) -> None:
        env = gui_snapshot(self.env_vars)
        _, buf = cv2.imencode(".jpg", frame)
        img_b64 = b64encode(buf).decode()
        ts = time.time()

        mode = env["mode"]
        if mode == "Bartender":
            self.status.set("Analyzing patron…")
            vec = [round(float(x), 6) for x in BioVector.from_frame(frame).arr]
            q_score = self.qtune.encode(vec)

            try:
                out1 = json.loads(await self.ai.chat(stage1_prompt(vec, q_score, self.cfg), img_b64, 400))
            except:
                out1 = {"score": q_score, "risk": "Safe"}

            try:
                out2 = json.loads(await self.ai.chat(stage2_prompt(out1, self.cfg), None, 100))
            except:
                decision = "Fail" if out1["risk"] == "Refuse" else "Pass"
                cd = 30 if decision == "Fail" else (15 if out1["risk"] == "Warn" else 0)
                out2 = {"decision": decision, "cooldown": cd}

            try:
                out3 = json.loads(await self.ai.chat(stage3_prompt(out1, self.cfg), None, 150))
            except:
                msgs = {
                    "Safe": "Enjoy your drink! 🙂",
                    "Warn": f"Estimated score {out1['score']:.2f}. Please slow down.",
                    "Refuse": f"Refused: {out1['score']:.2f} ≥ {self.cfg.intox_threshold:.2f}."
                }
                out3 = {"message": msgs[out1["risk"]]}

            record = {"ts": ts, "env": env, "out1": out1, "out2": out2, "out3": out3}
            await self.db.save(ts, record)
            self.status.set(f"Decision: {out2['decision']}")

            if self.cfg.mode_autonomous and out2["decision"] == "Fail":
                mb.showwarning("QOPS ALERT", "Service refused for intoxication.")

        else:  # Drug Abuser mode
            substance = env["substance"]
            self.status.set(f"Analyzing {substance} purity…")
            try:
                out_sub = json.loads(await self.ai.chat(stage1_substance_prompt(substance, self.cfg), img_b64, 400))
            except:
                out_sub = {"purity": 0.0, "contaminants": [], "risk": "Unknown"}

            message = (
                f"Purity: {out_sub['purity']:.1f}% | "
                f"Contaminants: {', '.join(out_sub['contaminants']) or 'None'} | "
                f"Risk: {out_sub['risk']}"
            )
            record = {"ts": ts, "env": env, "out_sub": out_sub}
            await self.db.save(ts, record)
            self.status.set(message)
            mb.showinfo("Purity Report", message)

    def stop(self) -> None:
        self.stop_ev.set()


# ════════════════════════════════════════════════════════════════
# TKINTER GUI APP
class QOPSApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("QOPS – Quantum Overdose & Purity Scanner")
        self.geometry("700x480")

        self.crypto   = AESGCMCrypto(MASTER_KEY)
        self.settings = POSSettings.load(self.crypto)
        if not os.path.exists(SETTINGS_FILE):
            self.settings.prompt_gui()
            self.settings.save(self.crypto)
        if not self.settings.api_key:
            mb.showerror("Missing API Key", "Set OpenAI key in Settings.")
            self.destroy()
            return

        self.status = tk.StringVar(value="Idle")
        tk.Label(self, textvariable=self.status, font=("Helvetica", 14)).pack(pady=8)

        env_frame = tk.Frame(self)
        env_frame.pack(padx=10, pady=6)

        # Crowd
        tk.Label(env_frame, text="Crowd").grid(row=0, column=0, sticky="e")
        crowd_var = tk.StringVar(value="normal")
        tk.Entry(env_frame, textvariable=crowd_var, width=12).grid(row=0, column=1, padx=6)

        # Mode selector
        tk.Label(env_frame, text="Mode").grid(row=0, column=2, sticky="e")
        mode_var = tk.StringVar(value="Bartender")
        tk.OptionMenu(env_frame, mode_var, "Bartender", "Drug Abuser").grid(row=0, column=3, padx=6)

        # Substance selector
        tk.Label(env_frame, text="Substance").grid(row=0, column=4, sticky="e")
        substance_var = tk.StringVar(value="Heroin")
        tk.OptionMenu(env_frame, substance_var, "Heroin", "Fentanyl", "Cocaine", "Methamphetamine").grid(row=0, column=5, padx=6)

        # Buttons
        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=6)
        tk.Button(btn_frame, text="Settings", command=self.open_settings).grid(row=0, column=0, padx=6)
        tk.Button(btn_frame, text="View Logs", command=self.view_logs).grid(row=0, column=1, padx=6)

        self.text = tk.Text(self, height=12, width=85, wrap="word")
        self.text.pack(padx=10, pady=6)

        env_vars = {"crowd": crowd_var, "mode": mode_var, "substance": substance_var}
        self.db      = ReportDB(self.settings.db_path, self.crypto)
        self.ai      = OpenAIClient(api_key=self.settings.api_key)
        self.scanner = ScannerThread(self.settings, self.db, self.ai, self.status, env_vars)
        self.scanner.start()

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def open_settings(self) -> None:
        self.settings.prompt_gui()
        self.settings.save(self.crypto)
        mb.showinfo("Saved", "Restart to apply changes.")

    def view_logs(self) -> None:
        rows = asyncio.run(self.db.list_reports())
        if not rows:
            mb.showinfo("Logs", "No records.")
            return
        opts = "\n".join(
            f"{rid} – {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))}"
            for rid, ts in rows[:20]
        )
        sel = sd.askstring("Select Log ID", opts)
        self.text.delete("1.0", tk.END)
        if sel:
            try:
                rid = int(sel.split()[0])
                rpt = asyncio.run(self.db.load(rid))
                self.text.insert(tk.END, json.dumps(rpt, indent=2))
            except Exception as e:
                mb.showerror("Error", str(e))

    def on_close(self) -> None:
        self.scanner.stop()
        self.destroy()


if __name__ == "__main__":
    try:
        QOPSApp().mainloop()
    except KeyboardInterrupt:
        LOGGER.info("Exiting QOPS.")

