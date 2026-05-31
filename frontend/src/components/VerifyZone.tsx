import { useState, useRef } from 'react';
import { UploadCloud, BarChart3, Loader2, AlertCircle } from 'lucide-react';
import { motion } from 'framer-motion';
import axios from 'axios';

interface VerifyMetrics {
    mse: number;
    psnr: number;
    ssim: number;
    patch_psnr_mean: number;
    patch_psnr_min: number;
    patch_ssim_mean: number;
    patch_ssim_min: number;
    resolution: string;
}

export default function VerifyZone() {
    const [origFile, setOrigFile] = useState<File | null>(null);
    const [reconFile, setReconFile] = useState<File | null>(null);
    const [status, setStatus] = useState<'idle' | 'processing' | 'success' | 'error'>('idle');
    const [metrics, setMetrics] = useState<VerifyMetrics | null>(null);
    const [heatmapSrc, setHeatmapSrc] = useState<string>('');
    const [errorMsg, setErrorMsg] = useState('');
    const origInputRef = useRef<HTMLInputElement>(null);
    const reconInputRef = useRef<HTMLInputElement>(null);

    const handleVerify = async () => {
        if (!origFile || !reconFile) return;
        setStatus('processing');
        setErrorMsg('');

        const formData = new FormData();
        formData.append('original', origFile);
        formData.append('reconstructed', reconFile);

        try {
            const res = await axios.post('http://localhost:5000/api/v1/verify', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
                timeout: 120000,
            });
            if (res.data.status === 'success') {
                setMetrics(res.data.metrics);
                setHeatmapSrc(`data:${res.data.heatmap_mime};base64,${res.data.heatmap_data}`);
                setStatus('success');
            } else {
                setErrorMsg(res.data.message || 'Verification failed');
                setStatus('error');
            }
        } catch (err: any) {
            setErrorMsg(err.response?.data?.message || err.message || 'Connection failed');
            setStatus('error');
        }
    };

    const getQualityLabel = (ssim: number) => {
        if (ssim >= 0.98) return { text: 'Excellent', color: 'text-emerald-400' };
        if (ssim >= 0.95) return { text: 'Very Good', color: 'text-emerald-400' };
        if (ssim >= 0.90) return { text: 'Good', color: 'text-yellow-400' };
        if (ssim >= 0.80) return { text: 'Fair', color: 'text-orange-400' };
        return { text: 'Poor', color: 'text-red-400' };
    };

    return (
        <div className="w-full h-full bg-bg-surface/50 backdrop-blur-[2px] border border-border-subtle rounded-2xl p-7 flex flex-col">
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h2 className="text-xl font-semibold text-text-primary tracking-tight">Verify Reconstruction</h2>
                    <p className="text-sm text-text-secondary mt-0.5">Compare original vs reconstructed image quality</p>
                </div>
                <span className="text-[11px] px-2.5 py-1 rounded-full bg-violet-500/10 text-violet-400 border border-violet-500/20"
                    style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                    SSIM · PSNR
                </span>
            </div>

            {status !== 'success' && (
                <>
                    {/* Dual Upload Zone */}
                    <div className="grid grid-cols-2 gap-4 mb-5">
                        {/* Original Image */}
                        <div
                            onClick={() => origInputRef.current?.click()}
                            className={`border-2 border-dashed rounded-xl p-5 text-center cursor-pointer transition-all duration-300 ${origFile
                                ? 'border-emerald-500/30 bg-emerald-500/5'
                                : 'border-border-subtle hover:border-text-tertiary bg-bg-elevated/30'
                                }`}
                        >
                            <UploadCloud className={`w-6 h-6 mx-auto mb-2 ${origFile ? 'text-emerald-400' : 'text-text-tertiary'}`} />
                            <p className="text-[12px] font-medium text-text-primary">
                                {origFile ? origFile.name : 'Original Image'}
                            </p>
                            <p className="text-[10px] text-text-tertiary mt-1">
                                {origFile ? `${(origFile.size / 1024 / 1024).toFixed(2)} MB` : 'Click to select'}
                            </p>
                            <input ref={origInputRef} type="file" accept="image/*" className="hidden"
                                onChange={(e) => setOrigFile(e.target.files?.[0] || null)} />
                        </div>

                        {/* Reconstructed Image */}
                        <div
                            onClick={() => reconInputRef.current?.click()}
                            className={`border-2 border-dashed rounded-xl p-5 text-center cursor-pointer transition-all duration-300 ${reconFile
                                ? 'border-violet-500/30 bg-violet-500/5'
                                : 'border-border-subtle hover:border-text-tertiary bg-bg-elevated/30'
                                }`}
                        >
                            <UploadCloud className={`w-6 h-6 mx-auto mb-2 ${reconFile ? 'text-violet-400' : 'text-text-tertiary'}`} />
                            <p className="text-[12px] font-medium text-text-primary">
                                {reconFile ? reconFile.name : 'Reconstructed Image'}
                            </p>
                            <p className="text-[10px] text-text-tertiary mt-1">
                                {reconFile ? `${(reconFile.size / 1024 / 1024).toFixed(2)} MB` : 'Click to select'}
                            </p>
                            <input ref={reconInputRef} type="file" accept="image/*" className="hidden"
                                onChange={(e) => setReconFile(e.target.files?.[0] || null)} />
                        </div>
                    </div>

                    {errorMsg && (
                        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-red-500/10 border border-red-500/20 mb-4">
                            <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
                            <p className="text-[11px] text-red-400">{errorMsg}</p>
                        </div>
                    )}

                    <button
                        onClick={handleVerify}
                        disabled={!origFile || !reconFile || status === 'processing'}
                        className={`w-full py-3 rounded-xl text-[13px] font-semibold transition-all duration-300 flex items-center justify-center gap-2 ${origFile && reconFile
                            ? 'bg-gradient-to-r from-violet-600 to-indigo-600 text-white hover:shadow-[0_0_24px_rgba(139,92,246,0.2)] hover:scale-[1.01]'
                            : 'bg-bg-elevated text-text-tertiary cursor-not-allowed'
                            }`}
                    >
                        {status === 'processing' ? (
                            <><Loader2 className="w-4 h-4 animate-spin" /> Analyzing...</>
                        ) : (
                            <><BarChart3 className="w-4 h-4" /> Run Verification</>
                        )}
                    </button>
                </>
            )}

            {/* Results */}
            {status === 'success' && metrics && (
                <motion.div
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex-1 flex flex-col gap-4 overflow-y-auto"
                >
                    {/* Quality Badge */}
                    <div className="flex items-center justify-between">
                        <span className={`text-lg font-bold ${getQualityLabel(metrics.ssim).color}`}>
                            {getQualityLabel(metrics.ssim).text} Quality
                        </span>
                        <span className="text-[11px] text-text-tertiary" style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                            {metrics.resolution}
                        </span>
                    </div>

                    {/* Global Metrics Grid */}
                    <div className="grid grid-cols-3 gap-3">
                        <div className="bg-bg-elevated/60 rounded-xl p-3.5 border border-border-subtle text-center">
                            <p className="text-[10px] text-text-tertiary uppercase tracking-wide mb-1" style={{ fontFamily: "'JetBrains Mono', monospace" }}>SSIM</p>
                            <p className="text-lg font-bold text-violet-400">{metrics.ssim.toFixed(4)}</p>
                        </div>
                        <div className="bg-bg-elevated/60 rounded-xl p-3.5 border border-border-subtle text-center">
                            <p className="text-[10px] text-text-tertiary uppercase tracking-wide mb-1" style={{ fontFamily: "'JetBrains Mono', monospace" }}>PSNR</p>
                            <p className="text-lg font-bold text-accent-cool">{metrics.psnr.toFixed(2)} <span className="text-[10px] text-text-tertiary">dB</span></p>
                        </div>
                        <div className="bg-bg-elevated/60 rounded-xl p-3.5 border border-border-subtle text-center">
                            <p className="text-[10px] text-text-tertiary uppercase tracking-wide mb-1" style={{ fontFamily: "'JetBrains Mono', monospace" }}>MSE</p>
                            <p className="text-lg font-bold text-accent-warm">{metrics.mse.toFixed(6)}</p>
                        </div>
                    </div>

                    {/* Patch Metrics */}
                    <div className="grid grid-cols-2 gap-3">
                        <div className="bg-bg-elevated/40 rounded-xl p-3 border border-border-subtle">
                            <p className="text-[10px] text-text-tertiary mb-2" style={{ fontFamily: "'JetBrains Mono', monospace" }}>PATCH PSNR</p>
                            <div className="flex justify-between text-[12px]">
                                <span className="text-text-secondary">Mean</span>
                                <span className="text-text-primary font-medium">{metrics.patch_psnr_mean.toFixed(2)} dB</span>
                            </div>
                            <div className="flex justify-between text-[12px] mt-1">
                                <span className="text-text-secondary">Min</span>
                                <span className="text-text-primary font-medium">{metrics.patch_psnr_min.toFixed(2)} dB</span>
                            </div>
                        </div>
                        <div className="bg-bg-elevated/40 rounded-xl p-3 border border-border-subtle">
                            <p className="text-[10px] text-text-tertiary mb-2" style={{ fontFamily: "'JetBrains Mono', monospace" }}>PATCH SSIM</p>
                            <div className="flex justify-between text-[12px]">
                                <span className="text-text-secondary">Mean</span>
                                <span className="text-text-primary font-medium">{metrics.patch_ssim_mean.toFixed(4)}</span>
                            </div>
                            <div className="flex justify-between text-[12px] mt-1">
                                <span className="text-text-secondary">Min</span>
                                <span className="text-text-primary font-medium">{metrics.patch_ssim_min.toFixed(4)}</span>
                            </div>
                        </div>
                    </div>

                    {/* Error Heatmap */}
                    {heatmapSrc && (
                        <div className="rounded-xl overflow-hidden border border-border-subtle">
                            <p className="text-[10px] text-text-tertiary uppercase tracking-wide px-3 pt-3 pb-2" style={{ fontFamily: "'JetBrains Mono', monospace" }}>ERROR HEATMAP</p>
                            <img src={heatmapSrc} alt="Error heatmap" className="w-full h-auto" />
                        </div>
                    )}

                    <button
                        onClick={() => { setStatus('idle'); setMetrics(null); setOrigFile(null); setReconFile(null); setHeatmapSrc(''); }}
                        className="text-[12px] text-text-tertiary hover:text-text-primary transition-colors py-2"
                    >
                        ← Run another comparison
                    </button>
                </motion.div>
            )}
        </div>
    );
}
