import { useState, useRef } from 'react';
import { UploadCloud, Download, Loader2 } from 'lucide-react';
import { motion } from 'framer-motion';
import axios from 'axios';

export default function DecompressorZone() {
    const [isDragging, setIsDragging] = useState(false);
    const [file, setFile] = useState<File | null>(null);
    const [isDecompressing, setIsDecompressing] = useState(false);
    const [success, setSuccess] = useState(false);
    const [imageData, setImageData] = useState<string | null>(null);
    const [downloadLink, setDownloadLink] = useState<string | null>(null);
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
            handleFileSelection(e.dataTransfer.files[0]);
        }
    };

    const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    };

    const handleFileSelection = (selectedFile: File) => {
        if (!selectedFile.name.toLowerCase().endsWith('.bin')) {
            alert('Invalid file. Only .bin files are supported.');
            return;
        }
        setFile(selectedFile);
        setSuccess(false);
        setImageData(null);
    };

    const handleDecompress = async () => {
        if (!file) return;

        setIsDecompressing(true);
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await axios.post('http://localhost:5000/api/v1/decompress', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
                timeout: 300000,
                maxBodyLength: Infinity,
                maxContentLength: Infinity,
            });

            const base64Data = `data:${response.data.mime_type};base64,${response.data.image_data}`;
            setImageData(base64Data);
            setDownloadLink(base64Data);
            setSuccess(true);
        } catch (error: any) {
            console.error('Decompression failed:', error);
            const msg = error.response?.data?.message || error.response?.data?.stderr || error.message || 'Unknown error';
            alert(`Reconstruction failed: ${msg}`);
        } finally {
            setIsDecompressing(false);
        }
    };

    return (
        <div className="w-full h-full bg-bg-surface/50 backdrop-blur-[2px] border border-border-subtle rounded-2xl p-7 flex flex-col">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h2 className="text-[17px] font-semibold text-text-primary tracking-tight">
                        Reconstruct Image
                    </h2>
                    <p className="text-text-secondary mt-0.5 text-[12px]">
                        Decode a <span className="text-accent-cool" style={{ fontFamily: "'JetBrains Mono', monospace" }}>.bin</span> payload back to the original image
                    </p>
                </div>
                <span className="px-2.5 py-1 rounded-md bg-bg-elevated border border-border-subtle text-[10px] text-text-secondary"
                    style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                    .BIN → IMG
                </span>
            </div>

            {/* Drop Zone */}
            {!success && !isDecompressing && (
                <div
                    onClick={() => fileInputRef.current?.click()}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    className={`w-full flex-1 rounded-xl p-10 flex flex-col items-center justify-center cursor-pointer transition-all duration-400 border-2 border-dashed group ${isDragging
                        ? 'border-accent-cool/50 bg-accent-cool/[0.04]'
                        : 'border-border-medium bg-bg-primary/40 hover:bg-bg-elevated/40 hover:border-accent-cool/20'
                        }`}
                >
                    <input
                        type="file"
                        ref={fileInputRef}
                        onChange={handleFileInput}
                        accept=".bin"
                        className="hidden"
                    />

                    {/* Upload Icon */}
                    <div className="relative mb-6">
                        <div className={`absolute inset-0 rounded-full transition-all duration-500 ${isDragging ? 'bg-accent-cool/10 scale-150' : 'bg-transparent scale-100'
                            }`} />
                        <div className={`w-16 h-16 rounded-2xl flex items-center justify-center transition-all duration-300 ${isDragging
                            ? 'bg-accent-cool/15 border border-accent-cool/30'
                            : 'bg-bg-elevated border border-border-medium group-hover:border-accent-cool/20'
                            }`}>
                            <UploadCloud className={`w-7 h-7 transition-colors duration-300 ${isDragging ? 'text-accent-cool' : 'text-text-tertiary group-hover:text-text-secondary'
                                }`} />
                        </div>
                    </div>

                    {file ? (
                        <div className="text-center">
                            <p className="text-[15px] font-medium text-text-primary">{file.name}</p>
                            <p className="text-[12px] text-accent-cool mt-1.5" style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                                Binary payload ready
                            </p>
                        </div>
                    ) : (
                        <div className="text-center">
                            <p className="text-[14px] font-medium text-text-primary mb-1">
                                Drop your .bin file here
                            </p>
                            <p className="text-[12px] text-text-tertiary">
                                or click to browse files
                            </p>
                        </div>
                    )}
                </div>
            )}

            {/* Action Button */}
            {file && !success && !isDecompressing && (
                <motion.button
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                    onClick={handleDecompress}
                    className="mt-5 w-full py-3.5 rounded-xl font-semibold text-[13px] tracking-wide transition-all duration-300
                               bg-gradient-to-r from-accent-cool to-accent-cool-muted text-white
                               hover:shadow-[0_4px_24px_rgba(126,200,227,0.25)] hover:-translate-y-0.5
                               active:translate-y-0 active:shadow-none"
                >
                    Reconstruct Image
                </motion.button>
            )}

            {/* Processing State */}
            {isDecompressing && (
                <div className="w-full flex-1 flex flex-col items-center justify-center py-16">
                    <div className="relative mb-5">
                        <Loader2 className="w-12 h-12 text-accent-cool animate-spin" />
                    </div>
                    <p className="text-text-secondary text-[12px] tracking-wider uppercase"
                        style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                        Reconstructing...
                    </p>
                </div>
            )}

            {/* Success State — Image Preview */}
            {success && imageData && (
                <motion.div
                    initial={{ opacity: 0, scale: 0.96 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
                    className="w-full flex flex-col items-center"
                >
                    {/* Image Preview Card */}
                    <div className="w-full relative rounded-xl overflow-hidden mb-5 border border-border-medium bg-bg-elevated/50">
                        <img src={imageData} alt="Reconstructed image" className="w-full h-auto max-h-[320px] object-cover" />
                        <div className="absolute bottom-2.5 left-2.5 px-2.5 py-1 bg-bg-primary/90 backdrop-blur-md rounded-lg border border-border-subtle flex items-center gap-2">
                            <span className="relative flex h-1.5 w-1.5">
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-60"></span>
                                <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-400"></span>
                            </span>
                            <span className="text-[10px] text-text-secondary" style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                                Verified
                            </span>
                        </div>
                    </div>

                    {/* Download Button */}
                    <a
                        href={downloadLink!}
                        download="neurosat_reconstruction.png"
                        className="w-full py-3.5 rounded-xl font-semibold text-[13px] tracking-wide transition-all duration-300
                                   bg-gradient-to-r from-accent-cool to-accent-cool-muted text-white
                                   hover:shadow-[0_4px_24px_rgba(126,200,227,0.25)] hover:-translate-y-0.5
                                   flex items-center justify-center gap-2"
                    >
                        <Download className="w-4 h-4" />
                        Save Image
                    </a>

                    <button
                        onClick={() => { setSuccess(false); setFile(null); setImageData(null); }}
                        className="mt-5 text-[12px] text-text-tertiary hover:text-text-primary transition-colors duration-200"
                    >
                        Reconstruct another file
                    </button>
                </motion.div>
            )}
        </div>
    );
}
