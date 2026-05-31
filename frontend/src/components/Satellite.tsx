import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

interface SatelliteConfig {
    orbitRadius: number;
    orbitSpeed: number;
    inclination: number;
    startAngle: number;
    scale: number;
    panelColor: string;
    signalColor: string;
}

/*
 * LaserBeam — computes the beam orientation EVERY FRAME using
 * world-space positions. This avoids all coordinate-transform bugs
 * from nested lookAt + rotation hierarchies.
 */
function LaserBeam({ satelliteGroupRef, color }: {
    satelliteGroupRef: React.RefObject<THREE.Group | null>;
    color: string;
}) {
    const beamGroupRef = useRef<THREE.Group>(null);
    const beamCoreRef = useRef<THREE.Mesh>(null);
    const beamGlowRef = useRef<THREE.Mesh>(null);
    const impactRef = useRef<THREE.Mesh>(null);
    const impactRingRef = useRef<THREE.Mesh>(null);

    const scanTimeRef = useRef(0);
    const scanActiveRef = useRef(false);
    const nextScanRef = useRef(Math.random() * 10 + 15);

    // Earth surface radius (the Earth sphere has radius ~4)
    const EARTH_RADIUS = 4.15;

    useFrame((_, delta) => {
        scanTimeRef.current += delta;

        if (!scanActiveRef.current && scanTimeRef.current > nextScanRef.current) {
            scanActiveRef.current = true;
            scanTimeRef.current = 0;
        }

        const active = scanActiveRef.current && scanTimeRef.current < 3.5;

        if (beamGroupRef.current) {
            beamGroupRef.current.visible = active;
        }

        if (!active) {
            if (scanActiveRef.current && scanTimeRef.current >= 3.5) {
                scanActiveRef.current = false;
                scanTimeRef.current = 0;
                nextScanRef.current = Math.random() * 15 + 20;
            }
            return;
        }

        // Get satellite world position
        if (!satelliteGroupRef.current || !beamGroupRef.current) return;
        const satWorldPos = new THREE.Vector3();
        satelliteGroupRef.current.getWorldPosition(satWorldPos);

        // Earth center in world space is wherever the parent group is
        // The Earth group is at [-2, -2, 0] in EarthScene.tsx
        const earthCenter = new THREE.Vector3(-2, -2, 0);

        // Direction from satellite to Earth center
        const dir = earthCenter.clone().sub(satWorldPos).normalize();

        // Beam endpoint: from satellite to just above Earth surface
        const distToSurface = satWorldPos.distanceTo(earthCenter) - EARTH_RADIUS;
        const beamLength = Math.max(distToSurface, 0.5);

        // Position beam group at satellite world position
        beamGroupRef.current.position.copy(satWorldPos);

        // Orient beam group to point toward Earth
        const lookTarget = satWorldPos.clone().add(dir);
        beamGroupRef.current.lookAt(lookTarget);

        // Now the beam group's -Z axis points toward Earth
        // Place children along -Z in local space

        const t = scanTimeRef.current;

        // Fade in/out
        let opacity: number;
        if (t < 0.3) opacity = t / 0.3;
        else if (t < 3.0) opacity = 1.0;
        else opacity = Math.max(0, 1 - (t - 3.0) / 0.5);

        // Core beam
        if (beamCoreRef.current) {
            beamCoreRef.current.position.set(0, 0, -beamLength / 2);
            beamCoreRef.current.scale.set(1, 1, beamLength);
            (beamCoreRef.current.material as THREE.MeshBasicMaterial).opacity = opacity * 0.85;
        }

        // Glow
        if (beamGlowRef.current) {
            beamGlowRef.current.position.set(0, 0, -beamLength / 2);
            beamGlowRef.current.scale.set(1, 1, beamLength);
            (beamGlowRef.current.material as THREE.MeshBasicMaterial).opacity = opacity * 0.2;
        }

        // Impact dot on Earth surface
        if (impactRef.current) {
            impactRef.current.position.set(0, 0, -beamLength);
            const pulse = 1 + Math.sin(t * 12) * 0.3;
            impactRef.current.scale.setScalar(pulse);
            (impactRef.current.material as THREE.MeshBasicMaterial).opacity = opacity * 0.8;
        }

        // Impact ring
        if (impactRingRef.current) {
            impactRingRef.current.position.set(0, 0, -beamLength + 0.02);
            const ringPhase = (t * 2) % 1;
            const ringScale = 0.5 + ringPhase * 1.5;
            impactRingRef.current.scale.set(ringScale, ringScale, 1);
            (impactRingRef.current.material as THREE.MeshBasicMaterial).opacity = opacity * (1 - ringPhase) * 0.4;
        }
    });

    return (
        <group ref={beamGroupRef} visible={false}>
            {/* Core laser — unit-length cylinder along Z, scaled in useFrame */}
            <mesh ref={beamCoreRef} rotation={[Math.PI / 2, 0, 0]}>
                <cylinderGeometry args={[0.01, 0.01, 1, 4]} />
                <meshBasicMaterial color={color} transparent opacity={0} depthWrite={false} blending={THREE.AdditiveBlending} />
            </mesh>

            {/* Glow envelope */}
            <mesh ref={beamGlowRef} rotation={[Math.PI / 2, 0, 0]}>
                <cylinderGeometry args={[0.05, 0.03, 1, 6]} />
                <meshBasicMaterial color={color} transparent opacity={0} depthWrite={false} blending={THREE.AdditiveBlending} />
            </mesh>

            {/* Impact dot */}
            <mesh ref={impactRef}>
                <sphereGeometry args={[0.1, 8, 8]} />
                <meshBasicMaterial color={color} transparent opacity={0} depthWrite={false} blending={THREE.AdditiveBlending} />
            </mesh>

            {/* Impact ring */}
            <mesh ref={impactRingRef}>
                <ringGeometry args={[0.12, 0.18, 16]} />
                <meshBasicMaterial color={color} transparent opacity={0} side={THREE.DoubleSide} depthWrite={false} blending={THREE.AdditiveBlending} />
            </mesh>
        </group>
    );
}

// ─── Single Satellite ───
function SingleSatellite({ config }: { config: SatelliteConfig }) {
    const orbitGroupRef = useRef<THREE.Group>(null);
    const satelliteRef = useRef<THREE.Group>(null);
    const signalRef = useRef<THREE.Mesh>(null);

    const panelTexture = useMemo(() => {
        const canvas = document.createElement('canvas');
        canvas.width = 64;
        canvas.height = 32;
        const ctx = canvas.getContext('2d')!;
        ctx.fillStyle = '#1a1a3e';
        ctx.fillRect(0, 0, 64, 32);
        ctx.strokeStyle = '#2a2a5e';
        ctx.lineWidth = 0.5;
        for (let x = 0; x < 64; x += 4) {
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, 32); ctx.stroke();
        }
        for (let y = 0; y < 32; y += 4) {
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(64, y); ctx.stroke();
        }
        return new THREE.CanvasTexture(canvas);
    }, []);

    useFrame((state, delta) => {
        if (orbitGroupRef.current) {
            orbitGroupRef.current.rotation.y += delta * config.orbitSpeed;
        }
        if (satelliteRef.current) {
            satelliteRef.current.lookAt(0, 0, 0);
            satelliteRef.current.rotation.z += Math.sin(state.clock.elapsedTime * 0.5) * 0.001;
        }
        if (signalRef.current) {
            const pulse = 0.4 + Math.sin(state.clock.elapsedTime * 3 + config.startAngle) * 0.3;
            (signalRef.current.material as THREE.MeshBasicMaterial).opacity = pulse;
        }
    });

    return (
        <group rotation={[config.inclination, 0, 0]}>
            <group ref={orbitGroupRef} rotation={[0, config.startAngle, 0]}>
                <group position={[config.orbitRadius, 0, 0]} ref={satelliteRef}>
                    <group scale={config.scale}>
                        <mesh><boxGeometry args={[0.4, 0.4, 0.7]} /><meshStandardMaterial color="#c0c0cc" metalness={0.8} roughness={0.3} /></mesh>
                        <mesh position={[0, 0, 0.351]}><planeGeometry args={[0.35, 0.35]} /><meshStandardMaterial color="#2a2a35" metalness={0.5} roughness={0.6} /></mesh>
                        <mesh position={[-0.55, 0, 0]}><cylinderGeometry args={[0.015, 0.015, 0.6, 6]} /><meshStandardMaterial color="#888899" metalness={0.7} roughness={0.4} /></mesh>
                        <mesh position={[-1.1, 0, 0]} rotation={[0, 0, Math.PI / 2]}><boxGeometry args={[0.5, 1.3, 0.025]} /><meshStandardMaterial map={panelTexture} metalness={0.3} roughness={0.5} color={config.panelColor} /></mesh>
                        <mesh position={[0.55, 0, 0]}><cylinderGeometry args={[0.015, 0.015, 0.6, 6]} /><meshStandardMaterial color="#888899" metalness={0.7} roughness={0.4} /></mesh>
                        <mesh position={[1.1, 0, 0]} rotation={[0, 0, Math.PI / 2]}><boxGeometry args={[0.5, 1.3, 0.025]} /><meshStandardMaterial map={panelTexture} metalness={0.3} roughness={0.5} color={config.panelColor} /></mesh>
                        <mesh position={[0, -0.35, -0.1]} rotation={[0.3, 0, 0]}><coneGeometry args={[0.2, 0.12, 12, 1, true]} /><meshStandardMaterial color="#e0e0e8" metalness={0.9} roughness={0.2} side={THREE.DoubleSide} /></mesh>
                        <mesh position={[0.08, 0.3, 0.08]}><cylinderGeometry args={[0.006, 0.006, 0.3, 4]} /><meshStandardMaterial color="#aaaabc" metalness={0.8} roughness={0.3} /></mesh>
                        <mesh ref={signalRef} position={[0, -0.38, -0.1]}><sphereGeometry args={[0.05, 8, 8]} /><meshBasicMaterial color={config.signalColor} transparent opacity={0.5} /></mesh>
                    </group>
                </group>
            </group>

            {/* Laser beam — rendered OUTSIDE of the satellite's lookAt group.
                 It reads the satellite's world position and computes beam independently. */}
            <LaserBeam satelliteGroupRef={satelliteRef} color={config.signalColor} />
        </group>
    );
}

const SATELLITES: SatelliteConfig[] = [
    { orbitRadius: 7, orbitSpeed: 0.21, inclination: 0.26, startAngle: 0, scale: 0.6, panelColor: '#4455aa', signalColor: '#E8A838' },
    { orbitRadius: 8.5, orbitSpeed: 0.14, inclination: -0.52, startAngle: Math.PI * 0.7, scale: 0.45, panelColor: '#3366aa', signalColor: '#7EC8E3' },
    { orbitRadius: 6.2, orbitSpeed: 0.28, inclination: 0.87, startAngle: Math.PI * 1.3, scale: 0.35, panelColor: '#5544aa', signalColor: '#bb88ee' },
];

export default function Satellite() {
    return (
        <group>
            {SATELLITES.map((config, i) => (
                <SingleSatellite key={i} config={config} />
            ))}
        </group>
    );
}
