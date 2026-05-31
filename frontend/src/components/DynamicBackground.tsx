import { useEffect, useRef } from 'react';

interface Particle {
    x: number;
    y: number;
    vx: number;
    vy: number;
    radius: number;
    color: string;
    alpha: number;
    baseAlpha: number;
    pulseSpeed: number;
    pulseOffset: number;
}

export default function DynamicBackground() {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const animFrameRef = useRef<number>(0);
    const particlesRef = useRef<Particle[]>([]);
    const mouseRef = useRef({ x: -1000, y: -1000 });

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d', { alpha: false, willReadFrequently: true });
        if (!ctx) return;

        const resize = () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        };
        resize();
        window.addEventListener('resize', resize);

        // Color palettes — warm ambers & cool blues, desaturated
        const warmColors = [
            'rgba(232, 168, 56,',   // amber gold
            'rgba(196, 138, 42,',   // dark amber
            'rgba(210, 155, 48,',   // mid amber
        ];
        const coolColors = [
            'rgba(126, 200, 227,',  // ice blue
            'rgba(90, 155, 181,',   // muted blue
            'rgba(100, 170, 200,',  // mid blue
        ];

        // Initialize particles
        const count = 35;
        const particles: Particle[] = [];
        for (let i = 0; i < count; i++) {
            const isWarm = i < count * 0.45;
            const colors = isWarm ? warmColors : coolColors;
            const baseAlpha = 0.08 + Math.random() * 0.12;
            particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                vx: (Math.random() - 0.5) * 0.3,
                vy: (Math.random() - 0.5) * 0.3,
                radius: 80 + Math.random() * 120,
                color: colors[Math.floor(Math.random() * colors.length)],
                alpha: baseAlpha,
                baseAlpha,
                pulseSpeed: 0.003 + Math.random() * 0.005,
                pulseOffset: Math.random() * Math.PI * 2,
            });
        }
        particlesRef.current = particles;

        // Mouse tracking for subtle interactivity
        const handleMouse = (e: MouseEvent) => {
            mouseRef.current = { x: e.clientX, y: e.clientY };
        };
        window.addEventListener('mousemove', handleMouse);

        let time = 0;
        const animate = () => {
            time++;
            ctx.fillStyle = '#060608';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Draw connection lines between nearby particles (very subtle)
            for (let i = 0; i < particles.length; i++) {
                for (let j = i + 1; j < particles.length; j++) {
                    const dx = particles[i].x - particles[j].x;
                    const dy = particles[i].y - particles[j].y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < 250) {
                        const lineAlpha = (1 - dist / 250) * 0.025;
                        ctx.beginPath();
                        ctx.strokeStyle = `rgba(255, 255, 255, ${lineAlpha})`;
                        ctx.lineWidth = 0.5;
                        ctx.moveTo(particles[i].x, particles[i].y);
                        ctx.lineTo(particles[j].x, particles[j].y);
                        ctx.stroke();
                    }
                }
            }

            // Draw and update particles
            for (const p of particles) {
                // Pulse alpha
                p.alpha = p.baseAlpha + Math.sin(time * p.pulseSpeed + p.pulseOffset) * 0.04;

                // Subtle mouse repulsion
                const mdx = p.x - mouseRef.current.x;
                const mdy = p.y - mouseRef.current.y;
                const mDist = Math.sqrt(mdx * mdx + mdy * mdy);
                if (mDist < 300 && mDist > 0) {
                    const force = (1 - mDist / 300) * 0.15;
                    p.vx += (mdx / mDist) * force;
                    p.vy += (mdy / mDist) * force;
                }

                // Dampen velocity
                p.vx *= 0.995;
                p.vy *= 0.995;

                // Move
                p.x += p.vx;
                p.y += p.vy;

                // Wrap around edges
                if (p.x < -p.radius) p.x = canvas.width + p.radius;
                if (p.x > canvas.width + p.radius) p.x = -p.radius;
                if (p.y < -p.radius) p.y = canvas.height + p.radius;
                if (p.y > canvas.height + p.radius) p.y = -p.radius;

                // Draw soft glow particle
                const gradient = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.radius);
                gradient.addColorStop(0, `${p.color} ${p.alpha})`);
                gradient.addColorStop(0.4, `${p.color} ${p.alpha * 0.4})`);
                gradient.addColorStop(1, `${p.color} 0)`);

                ctx.beginPath();
                ctx.fillStyle = gradient;
                ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
                ctx.fill();
            }

            // Vignette overlay
            const vignette = ctx.createRadialGradient(
                canvas.width / 2, canvas.height / 2, canvas.height * 0.2,
                canvas.width / 2, canvas.height / 2, canvas.height * 0.9
            );
            vignette.addColorStop(0, 'rgba(6, 6, 8, 0)');
            vignette.addColorStop(1, 'rgba(6, 6, 8, 0.7)');
            ctx.fillStyle = vignette;
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            animFrameRef.current = requestAnimationFrame(animate);
        };

        animate();

        return () => {
            window.removeEventListener('resize', resize);
            window.removeEventListener('mousemove', handleMouse);
            cancelAnimationFrame(animFrameRef.current);
        };
    }, []);

    return (
        <canvas
            ref={canvasRef}
            className="fixed inset-0 z-0"
            style={{ pointerEvents: 'none' }}
        />
    );
}
