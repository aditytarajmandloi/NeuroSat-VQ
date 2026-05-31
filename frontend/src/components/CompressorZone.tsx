import { useState, useRef } from 'react';
import { UploadCloud, CheckCircle, ArrowDownToLine, Loader2 } from 'lucide-react';
import { motion } from 'framer-motion';
import axios from 'axios';

interface CompressionMetrics {
    original_size_bytes: number;
    compressed_size_bytes: number;
    compression_ratio: string;
}

export default function CompressorZone() {
    const [isDragging, setIsDragging] = useState(false);
    const [file, setFile] = useState<File | null>(null);
    const [isCompressing, setIsCompressing] = useState(false);
    const [success, setSuccess] = useState(false);
    const [metrics, setMetrics] = useState<CompressionMetrics | null>(null);
    const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);

        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            const droppedFile = e.dataTransfer.files[0];
            handleFileSelection(droppedFile);
        }
    };

    const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    };

    const handleFileSelection = (selectedFile: File) => {
        const validTypes = ['image/png', 'image/jpeg', 'image/tiff', 'image/tif'];
        const validExts = ['.png', '.jpg', '.jpeg', '.tiff', '.tif'];
        const isValid = validTypes.includes(selectedFile.type) || validExts.some(ext => selectedFile.name.toLowerCase().endsWith(ext));
        if (!isValid) {
            alert('Invalid file format. Please upload PNG, JPG, or TIFF.');
            return;
        }
        setFile(selectedFile);
        setSuccess(false);
        setMetrics(null);
    };

    const handleCompress = async () => {
        if (!file) return;

        setIsCompressing(true);
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await axios.post('http://localhost:5000/api/v1/compress', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
                timeout: 300000, // 5 min for large images
                maxBodyLength: Infinity,
                maxContentLength: Infinity,
            });

            setMetrics(response.data.metrics);
            setDownloadUrl(`http://localhost:5000${response.data.download_url}`);
            setSuccess(true);
        } catch (error: any) {
            console.error('Compression failed:', error);
            const msg = error.response?.data?.message || error.response?.data?.stderr || error.message || 'Unknown error';
            alert(`Compression failed: ${msg}`);
        } finally {
            setIsCompressing(false);
        }
    };

    const formatBytes = (bytes: number, decimals = 2) => {
        if (!+bytes) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
    };

    return (
        <div className="w-full h-full bg-bg-surface/50 backdrop-blur-[2px] border border-border-subtle rounded-2xl p-7 flex flex-col">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h2 className="text-[17px] font-semibold text-text-primary tracking-tight">
                        Compress Image
                    </h2>
                    <p className="text-text-secondary mt-0.5 text-[12px]">
                        Encode high-res imagery into compact <span className="text-accent-warm" style={{ fontFamily: "'JetBrains Mono', monospace" }}>.bin</span> format
                    </p>
                </div>
                <div className="flex gap-2">
                    {['PNG', 'JPG', 'TIFF'].map((fmt) => (
                        <span key={fmt} className="px-2 py-1 rounded-md bg-bg-elevated border border-border-subtle text-[10px] text-text-secondary"
                            style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                            {fmt}
                        </span>
                    ))}
                </div>
            </div>

            {/* Drop Zone */}
            {!success && !isCompressing && (
                <div
                    onClick={() => fileInputRef.current?.click()}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    className={`w-full flex-1 rounded-xl p-10 flex flex-col items-center justify-center cursor-pointer transition-all duration-400 border-2 border-dashed group ${isDragging
                        ? 'border-accent-warm/50 bg-accent-warm/[0.04]'
                        : 'border-border-medium bg-bg-primary/40 hover:bg-bg-elevated/40 hover:border-accent-warm/20'
                        }`}
                >
                    <input
                        type="file"
                        ref={fileInputRef}
                        onChange={handleFileInput}
                        accept=".png,.jpg,.jpeg,.tiff,.tif"
                        className="hidden"
                    />

                    {/* Upload Icon with Pulse Ring */}
                    <div className="relative mb-6">
                        <div className={`absolute inset-0 rounded-full transition-all duration-500 ${isDragging ? 'bg-accent-warm/10 scale-150' : 'bg-transparent scale-100'
                            }`} />
                        <div className={`w-16 h-16 rounded-2xl flex items-center justify-center transition-all duration-300 ${isDragging
                            ? 'bg-accent-warm/15 border border-accent-warm/30'
                            : 'bg-bg-elevated border border-border-medium group-hover:border-accent-warm/20'
                            }`}>
                            <UploadCloud className={`w-7 h-7 transition-colors duration-300 ${isDragging ? 'text-accent-warm' : 'text-text-tertiary group-hover:text-text-secondary'
                                }`} />
                        </div>
                    </div>

                    {file ? (
                        <div className="text-center">
                            <p className="text-[15px] font-medium text-text-primary">{file.name}</p>
                            <p className="text-[12px] text-text-secondary mt-1.5" style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                                {formatBytes(file.size)}
                            </p>
                        </div>
                    ) : (
                        <div className="text-center">
                            <p className="text-[14px] font-medium text-text-primary mb-1">
                                Drop your image here
                            </p>
                            <p className="text-[12px] text-text-tertiary">
                                or click to browse files
                            </p>
                        </div>
                    )}
                </div>
            )}

            {/* Action Button */}
            {file && !success && !isCompressing && (
                <motion.button
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                    onClick={handleCompress}
                    className="mt-5 w-full py-3.5 rounded-xl font-semibold text-[13px] tracking-wide transition-all duration-300
                               bg-gradient-to-r from-accent-warm to-accent-warm-muted text-white
                               hover:shadow-[0_4px_24px_rgba(232,168,56,0.25)] hover:-translate-y-0.5
                               active:translate-y-0 active:shadow-none"
                >
                    Compress Image
                </motion.button>
            )}

            {/* Processing State */}
            {isCompressing && (
                <div className="w-full flex-1 flex flex-col items-center justify-center py-16">
                    <div className="relative mb-5">
                        <div className="absolute inset-0 rounded-full bg-accent-warm/10 animate-pulse-ring" style={{ width: 56, height: 56, margin: '-4px' }} />
                        <Loader2 className="w-12 h-12 text-accent-warm animate-spin" />
                    </div>
                    <p className="text-text-secondary text-[12px] tracking-wider uppercase"
                        style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                        Compressing...
                    </p>
                </div>
            )}

            {/* Success State */}
            {success && metrics && (
                <motion.div
                    initial={{ opacity: 0, scale: 0.96 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
                    className="w-full flex flex-col items-center"
                >
                    {/* Success Badge */}
                    <div className="w-14 h-14 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center mb-5">
                        <CheckCircle className="w-7 h-7 text-emerald-400" />
                    </div>
                    <h3 className="text-[16px] font-semibold text-text-primary mb-1">Done</h3>
                    <p className="text-[12px] text-text-secondary mb-6">Image compressed successfully</p>

                    {/* Metrics Grid */}
                    <div className="w-full grid grid-cols-3 gap-3 mb-6">
                        <div className="bg-bg-elevated/80 rounded-xl p-4 border border-border-subtle text-center">
                            <span className="text-[10px] text-text-tertiary uppercase tracking-wider block mb-1.5"
                                style={{ fontFamily: "'JetBrains Mono', monospace" }}>Original</span>
                            <span className="text-[15px] font-semibold text-text-primary" style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                                {formatBytes(metrics.original_size_bytes)}
                            </span>
                        </div>
                        <div className="bg-bg-elevated/80 rounded-xl p-4 border border-border-subtle text-center">
                            <span className="text-[10px] text-text-tertiary uppercase tracking-wider block mb-1.5"
                                style={{ fontFamily: "'JetBrains Mono', monospace" }}>Compressed</span>
                            <span className="text-[15px] font-semibold text-accent-warm" style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                                {formatBytes(metrics.compressed_size_bytes)}
                            </span>
                        </div>
                        <div className="bg-accent-warm/[0.06] rounded-xl p-4 border border-accent-warm/15 text-center">
                            <span className="text-[10px] text-accent-warm/70 uppercase tracking-wider block mb-1.5"
                                style={{ fontFamily: "'JetBrains Mono', monospace" }}>Ratio</span>
                            <span className="text-[18px] font-bold text-accent-warm" style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                                {metrics.compression_ratio}
                            </span>
                        </div>
                    </div>

                    {/* Download Button */}
                    <a
                        href={downloadUrl!}
                        className="w-full py-3.5 rounded-xl font-semibold text-[13px] tracking-wide transition-all duration-300
                                   bg-gradient-to-r from-accent-warm to-accent-warm-muted text-white
                                   hover:shadow-[0_4px_24px_rgba(232,168,56,0.25)] hover:-translate-y-0.5
                                   flex items-center justify-center gap-2"
                    >
                        <ArrowDownToLine className="w-4 h-4" />
                        Download .bin
                    </a>

                    <button
                        onClick={() => { setSuccess(false); setFile(null); }}
                        className="mt-5 text-[12px] text-text-tertiary hover:text-text-primary transition-colors duration-200"
                    >
                        Compress another image
                    </button>
                </motion.div>
            )}
        </div>
    );
}
