import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

/*
 * Volumetric sun rays — simulated with multiple semi-transparent
 * tapered planes arranged as a fan emanating from the sun position.
 * Each "ray" is a long, narrow plane with additive blending that
 * slowly pulses and shifts, creating a god-ray illusion.
 */

const SUN_POSITION = new THREE.Vector3(30, 15, 10);
const RAY_COUNT = 8;

interface RayData {
    angle: number;
    length: number;
    width: number;
    opacity: number;
    phaseOffset: number;
}

export default function SunRays() {
    const groupRef = useRef<THREE.Group>(null);
    const rayRefs = useRef<(THREE.Mesh | null)[]>([]);

    // Generate ray configurations
    const rays = useMemo<RayData[]>(() => {
        const arr: RayData[] = [];
        for (let i = 0; i < RAY_COUNT; i++) {
            arr.push({
                angle: (i / RAY_COUNT) * Math.PI * 0.4 - Math.PI * 0.2 + (Math.random() - 0.5) * 0.1,
                length: 25 + Math.random() * 15,
                width: 0.4 + Math.random() * 0.6,
                opacity: 0.02 + Math.random() * 0.03,
                phaseOffset: Math.random() * Math.PI * 2,
            });
        }
        return arr;
    }, []);

    // Sun glow texture (procedural radial gradient)
    const sunGlowTexture = useMemo(() => {
        const size = 128;
        const canvas = document.createElement('canvas');
        canvas.width = size;
        canvas.height = size;
        const ctx = canvas.getContext('2d')!;
        const gradient = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
        gradient.addColorStop(0, 'rgba(255, 248, 230, 1)');
        gradient.addColorStop(0.15, 'rgba(255, 230, 180, 0.8)');
        gradient.addColorStop(0.4, 'rgba(255, 200, 120, 0.3)');
        gradient.addColorStop(0.7, 'rgba(255, 180, 80, 0.1)');
        gradient.addColorStop(1, 'rgba(255, 160, 60, 0)');
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, size, size);
        return new THREE.CanvasTexture(canvas);
    }, []);

    // Ray gradient texture
    const rayTexture = useMemo(() => {
        const canvas = document.createElement('canvas');
        canvas.width = 256;
        canvas.height = 32;
        const ctx = canvas.getContext('2d')!;
        const gradient = ctx.createLinearGradient(0, 0, 256, 0);
        gradient.addColorStop(0, 'rgba(255, 240, 200, 0.8)');
        gradient.addColorStop(0.2, 'rgba(255, 220, 160, 0.4)');
        gradient.addColorStop(0.6, 'rgba(255, 200, 130, 0.15)');
        gradient.addColorStop(1, 'rgba(255, 180, 100, 0)');
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, 256, 32);

        // Vertical gradient — fade at edges
        const vGrad = ctx.createLinearGradient(0, 0, 0, 32);
        vGrad.addColorStop(0, 'rgba(0,0,0,0)');
        vGrad.addColorStop(0.3, 'rgba(255,255,255,1)');
        vGrad.addColorStop(0.7, 'rgba(255,255,255,1)');
        vGrad.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.globalCompositeOperation = 'destination-in';
        ctx.fillStyle = vGrad;
        ctx.fillRect(0, 0, 256, 32);

        return new THREE.CanvasTexture(canvas);
    }, []);

    useFrame((state) => {
        const time = state.clock.elapsedTime;

        // Subtle animation on each ray
        rays.forEach((ray, i) => {
            const mesh = rayRefs.current[i];
            if (!mesh) return;
            const mat = mesh.material as THREE.MeshBasicMaterial;
            // Slowly pulse opacity
            const pulse = 0.7 + Math.sin(time * 0.3 + ray.phaseOffset) * 0.3;
            mat.opacity = ray.opacity * pulse;
        });
    });

    return (
        <group ref={groupRef}>
            {/* Sun disc — bright point source */}
            <mesh position={SUN_POSITION.toArray()}>
                <sphereGeometry args={[1.5, 16, 16]} />
                <meshBasicMaterial
                    color="#FFF8E8"
                    transparent
                    opacity={0.95}
                />
            </mesh>

            {/* Sun corona glow — billboard sprite */}
            <mesh position={SUN_POSITION.toArray()}>
                <planeGeometry args={[12, 12]} />
                <meshBasicMaterial
                    map={sunGlowTexture}
                    transparent
                    opacity={0.5}
                    depthWrite={false}
                    blending={THREE.AdditiveBlending}
                />
            </mesh>

            {/* Secondary larger glow */}
            <mesh position={SUN_POSITION.toArray()}>
                <planeGeometry args={[25, 25]} />
                <meshBasicMaterial
                    map={sunGlowTexture}
                    transparent
                    opacity={0.12}
                    depthWrite={false}
                    blending={THREE.AdditiveBlending}
                />
            </mesh>

            {/* God rays — elongated planes emanating from the sun */}
            {rays.map((ray, i) => {
                // Position each ray starting at the sun and extending toward the scene
                const dir = new THREE.Vector3(-1, -0.4, -0.2).normalize();
                // Rotate direction by ray angle (fan spread)
                const axis = new THREE.Vector3(0, 0, 1);
                const rotatedDir = dir.clone().applyAxisAngle(axis, ray.angle);

                // Center of the ray plane: halfway along its length from the sun
                const center = SUN_POSITION.clone().add(rotatedDir.clone().multiplyScalar(ray.length / 2));

                // Orientation: plane faces perpendicular to camera (billboarded vertically)
                const quat = new THREE.Quaternion();
                const up = new THREE.Vector3(0, 1, 0);
                const lookDir = rotatedDir.clone();
                const mat4 = new THREE.Matrix4().lookAt(
                    new THREE.Vector3(0, 0, 0),
                    lookDir,
                    up
                );
                quat.setFromRotationMatrix(mat4);

                return (
                    <mesh
                        key={i}
                        ref={(el) => { rayRefs.current[i] = el; }}
                        position={center.toArray()}
                        quaternion={quat.toArray() as [number, number, number, number]}
                    >
                        <planeGeometry args={[ray.width, ray.length]} />
                        <meshBasicMaterial
                            map={rayTexture}
                            transparent
                            opacity={ray.opacity}
                            side={THREE.DoubleSide}
                            depthWrite={false}
                            blending={THREE.AdditiveBlending}
                        />
                    </mesh>
                );
            })}
        </group>
    );
}
