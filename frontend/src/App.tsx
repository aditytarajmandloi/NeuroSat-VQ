import { useState } from 'react';
import { Zap, Layers, BarChart3, Satellite, ChevronRight } from 'lucide-react';
import CompressorZone from './components/CompressorZone';
import DecompressorZone from './components/DecompressorZone';
import VerifyZone from './components/VerifyZone';
import DynamicBackground from './components/DynamicBackground';
import EarthScene from './components/EarthScene';
import { motion, AnimatePresence } from 'framer-motion';

type TabId = 'about' | 'compress' | 'decompress' | 'verify';

function App() {
    const [activeTab, setActiveTab] = useState<TabId>('about');

    const tabs: { id: TabId; label: string; icon: typeof Zap; accentClass: string; bgClass: string }[] = [
        { id: 'compress', label: 'Compress', icon: Zap, accentClass: 'text-accent-warm', bgClass: 'bg-accent-warm/10 border-accent-warm/20' },
        { id: 'decompress', label: 'Reconstruct', icon: Layers, accentClass: 'text-accent-cool', bgClass: 'bg-accent-cool/10 border-accent-cool/20' },
        { id: 'verify', label: 'Verify', icon: BarChart3, accentClass: 'text-violet-400', bgClass: 'bg-violet-500/10 border-violet-500/20' },
    ];

    return (
        <div className="min-h-screen w-full relative overflow-x-hidden" style={{ fontFamily: "'Inter', system-ui, sans-serif" }}>
            <DynamicBackground />
            <EarthScene />

            <div className="max-w-6xl mx-auto px-6 sm:px-8 py-8 relative z-10 min-h-screen flex flex-col">

                {/* ─── Header ─── */}
                <motion.header
                    initial={{ opacity: 0, y: -12 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
                    className="w-full flex justify-between items-center mb-10"
                >
                    <div className="flex items-center gap-3 cursor-pointer" onClick={() => setActiveTab('about')}>
                        {/* Redesigned Logo — Satellite dish icon with subtle gradient border */}
                        <div className="w-9 h-9 rounded-xl bg-bg-elevated border border-border-subtle flex items-center justify-center relative overflow-hidden group">
                            <div className="absolute inset-0 bg-gradient-to-br from-white/[0.04] to-transparent" />
                            <Satellite className="w-[18px] h-[18px] text-text-primary relative z-10 group-hover:text-accent-cool transition-colors duration-300" />
                        </div>
                        <div>
                            <h1 className="text-[15px] font-semibold tracking-tight text-text-primary leading-none">
                                NeuroSat
                            </h1>
                            <p className="text-[10px] text-text-tertiary tracking-wide mt-0.5" style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                                VQ Engine v2.7
                            </p>
                        </div>
                    </div>

                    <div className="flex items-center gap-3">
                        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-bg-elevated/80 border border-border-subtle">
                            <span className="relative flex h-1.5 w-1.5">
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-60"></span>
                                <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-400"></span>
                            </span>
                            <span className="text-[10px] text-text-secondary" style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                                Local
                            </span>
                        </div>
                    </div>
                </motion.header>

                {/* ─── Bento Grid ─── */}
                <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-5 w-full pb-8">

                    {/* Left Sidebar */}
                    <motion.aside
                        initial={{ opacity: 0, x: -16 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.6, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
                        className="lg:col-span-3 flex flex-col gap-4"
                    >
                        {/* Tab Controller */}
                        <div className="bg-bg-surface/80 backdrop-blur-sm border border-border-subtle rounded-2xl p-3">
                            <p className="text-[10px] font-medium text-text-tertiary uppercase tracking-widest px-2 mb-3"
                                style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                                Mode
                            </p>
                            <div className="flex flex-col gap-1.5">
                                {tabs.map((tab) => (
                                    <button
                                        key={tab.id}
                                        onClick={() => setActiveTab(tab.id)}
                                        className={`relative w-full px-3.5 py-2.5 rounded-xl text-[13px] font-medium transition-all duration-300 flex items-center gap-2.5 ${activeTab === tab.id
                                            ? `${tab.bgClass} ${tab.accentClass} border`
                                            : 'text-text-secondary hover:text-text-primary hover:bg-bg-hover border border-transparent'
                                            }`}
                                    >
                                        {activeTab === tab.id && (
                                            <motion.div
                                                layoutId="tab-glow"
                                                className="absolute inset-0 rounded-xl opacity-5"
                                                style={{ background: 'currentColor' }}
                                                transition={{ type: "spring", bounce: 0.2, duration: 0.5 }}
                                            />
                                        )}
                                        <tab.icon className="w-3.5 h-3.5 relative z-10" />
                                        <span className="relative z-10">{tab.label}</span>
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Info Card */}
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ duration: 0.8, delay: 0.3 }}
                            className="bg-bg-surface/60 backdrop-blur-sm border border-border-subtle rounded-2xl p-5 flex-1"
                        >
                            <p className="text-[10px] font-medium text-text-tertiary uppercase tracking-widest mb-4"
                                style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                                Engine
                            </p>
                            <div className="space-y-3.5">
                                <div className="flex justify-between items-center">
                                    <span className="text-[12px] text-text-secondary">Quantization</span>
                                    <span className="text-[11px] text-text-primary font-medium px-2 py-0.5 rounded-md bg-bg-elevated border border-border-subtle"
                                        style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                                        VQ-VAE
                                    </span>
                                </div>
                                <div className="flex justify-between items-center">
                                    <span className="text-[12px] text-text-secondary">Runtime</span>
                                    <span className="text-[11px] text-emerald-400 font-medium px-2 py-0.5 rounded-md bg-emerald-400/5 border border-emerald-400/10"
                                        style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                                        PyTorch
                                    </span>
                                </div>
                                <div className="flex justify-between items-center">
                                    <span className="text-[12px] text-text-secondary">Compute</span>
                                    <span className="text-[11px] text-text-primary font-medium px-2 py-0.5 rounded-md bg-bg-elevated border border-border-subtle"
                                        style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                                        Local
                                    </span>
                                </div>
                                <div className="flex justify-between items-center">
                                    <span className="text-[12px] text-text-secondary">Max Upload</span>
                                    <span className="text-[11px] text-text-primary font-medium px-2 py-0.5 rounded-md bg-bg-elevated border border-border-subtle"
                                        style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                                        50 MB
                                    </span>
                                </div>
                            </div>
                            <div className="mt-6 pt-4 border-t border-border-subtle">
                                <p className="text-[11px] text-text-tertiary leading-relaxed">
                                    All processing runs locally via PyTorch. No data leaves your machine.
                                </p>
                            </div>
                        </motion.div>
                    </motion.aside>

                    {/* Right Main Panel */}
                    <div className="lg:col-span-9 flex flex-col min-h-[500px]">
                        <AnimatePresence mode="wait">
                            {activeTab === 'about' && (
                                <motion.div
                                    key="about"
                                    initial={{ opacity: 0, y: 12 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, y: -12 }}
                                    transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
                                    className="h-full"
                                >
                                    {/* About / Landing View */}
                                    <div className="w-full h-full bg-bg-surface/50 backdrop-blur-[2px] border border-border-subtle rounded-2xl p-8 flex flex-col justify-between">
                                        <div>
                                            <motion.div
                                                initial={{ opacity: 0, y: 8 }}
                                                animate={{ opacity: 1, y: 0 }}
                                                transition={{ delay: 0.1 }}
                                            >
                                                <h2 className="text-2xl font-bold text-text-primary tracking-tight mb-2">
                                                    NeuroSat VQ Engine
                                                </h2>
                                                <p className="text-sm text-text-secondary mb-8 max-w-lg leading-relaxed">
                                                    A neural image compression system built on <strong className="text-text-primary">Vector Quantized Variational Autoencoders</strong> (VQ-VAE).
                                                    Compresses high-resolution imagery into compact binary payloads with minimal perceptual loss.
                                                </p>
                                            </motion.div>

                                            <motion.div
                                                initial={{ opacity: 0, y: 8 }}
                                                animate={{ opacity: 1, y: 0 }}
                                                transition={{ delay: 0.2 }}
                                                className="grid grid-cols-3 gap-4 mb-8"
                                            >
                                                <div className="bg-bg-elevated/50 rounded-xl p-4 border border-border-subtle">
                                                    <p className="text-[10px] text-text-tertiary uppercase tracking-widest mb-2" style={{ fontFamily: "'JetBrains Mono', monospace" }}>Architecture</p>
                                                    <p className="text-sm text-text-primary font-medium">VQ-VAE</p>
                                                    <p className="text-[11px] text-text-tertiary mt-1">Encoder → Codebook → Decoder</p>
                                                </div>
                                                <div className="bg-bg-elevated/50 rounded-xl p-4 border border-border-subtle">
                                                    <p className="text-[10px] text-text-tertiary uppercase tracking-widest mb-2" style={{ fontFamily: "'JetBrains Mono', monospace" }}>Codebook</p>
                                                    <p className="text-sm text-text-primary font-medium">512 Vectors</p>
                                                    <p className="text-[11px] text-text-tertiary mt-1">Discrete latent representations</p>
                                                </div>
                                                <div className="bg-bg-elevated/50 rounded-xl p-4 border border-border-subtle">
                                                    <p className="text-[10px] text-text-tertiary uppercase tracking-widest mb-2" style={{ fontFamily: "'JetBrains Mono', monospace" }}>Quality</p>
                                                    <p className="text-sm text-text-primary font-medium">SSIM ≥ 0.95</p>
                                                    <p className="text-[11px] text-text-tertiary mt-1">Near-lossless reconstruction</p>
                                                </div>
                                            </motion.div>

                                            <motion.div
                                                initial={{ opacity: 0, y: 8 }}
                                                animate={{ opacity: 1, y: 0 }}
                                                transition={{ delay: 0.3 }}
                                                className="space-y-3 mb-8"
                                            >
                                                <h3 className="text-xs font-medium text-text-tertiary uppercase tracking-widest" style={{ fontFamily: "'JetBrains Mono', monospace" }}>How it works</h3>
                                                <div className="flex items-center gap-3">
                                                    <div className="w-7 h-7 rounded-lg bg-accent-warm/10 flex items-center justify-center text-accent-warm text-xs font-bold flex-shrink-0">1</div>
                                                    <p className="text-[12px] text-text-secondary"><strong className="text-text-primary">Compress</strong> — Upload an image, the encoder maps it to discrete codebook indices stored as a compact <code className="text-accent-warm">.bin</code> file</p>
                                                </div>
                                                <div className="flex items-center gap-3">
                                                    <div className="w-7 h-7 rounded-lg bg-accent-cool/10 flex items-center justify-center text-accent-cool text-xs font-bold flex-shrink-0">2</div>
                                                    <p className="text-[12px] text-text-secondary"><strong className="text-text-primary">Reconstruct</strong> — Upload the <code className="text-accent-cool">.bin</code> file, the decoder reconstructs the original image from codebook vectors</p>
                                                </div>
                                                <div className="flex items-center gap-3">
                                                    <div className="w-7 h-7 rounded-lg bg-violet-500/10 flex items-center justify-center text-violet-400 text-xs font-bold flex-shrink-0">3</div>
                                                    <p className="text-[12px] text-text-secondary"><strong className="text-text-primary">Verify</strong> — Compare original vs. reconstructed with SSIM, PSNR, MSE metrics and error heatmaps</p>
                                                </div>
                                            </motion.div>
                                        </div>

                                        <motion.div
                                            initial={{ opacity: 0 }}
                                            animate={{ opacity: 1 }}
                                            transition={{ delay: 0.5 }}
                                        >
                                            <button
                                                onClick={() => setActiveTab('compress')}
                                                className="group flex items-center gap-2 text-[13px] font-semibold text-accent-warm hover:text-accent-warm transition-colors"
                                            >
                                                Start Compressing
                                                <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform duration-200" />
                                            </button>
                                        </motion.div>
                                    </div>
                                </motion.div>
                            )}

                            {activeTab === 'compress' && (
                                <motion.div
                                    key="compress"
                                    initial={{ opacity: 0, x: 12 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: -12 }}
                                    transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
                                    className="h-full"
                                >
                                    <CompressorZone />
                                </motion.div>
                            )}

                            {activeTab === 'decompress' && (
                                <motion.div
                                    key="decompress"
                                    initial={{ opacity: 0, x: 12 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: -12 }}
                                    transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
                                    className="h-full"
                                >
                                    <DecompressorZone />
                                </motion.div>
                            )}

                            {activeTab === 'verify' && (
                                <motion.div
                                    key="verify"
                                    initial={{ opacity: 0, x: 12 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: -12 }}
                                    transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
                                    className="h-full"
                                >
                                    <VerifyZone />
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>

                </div>

                {/* ─── Footer ─── */}
                <motion.footer
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 1, delay: 0.6 }}
                    className="w-full pt-4 pb-6 border-t border-border-subtle flex items-center justify-between"
                >
                    <span className="text-[11px] text-text-tertiary">
                        NeuroSat VQ v2.7
                    </span>
                    <span className="text-[10px] text-text-tertiary" style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                        Local session • All processing on-device
                    </span>
                </motion.footer>

            </div>
        </div>
    )
}

export default App
