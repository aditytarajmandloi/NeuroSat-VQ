import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

interface ShootingStar {
    startPos: THREE.Vector3;
    endPos: THREE.Vector3;
    speed: number;
    delay: number;
    length: number;
    brightness: number;
}

export default function LightTrails() {
    const trailsRef = useRef<THREE.Group>(null);
    const meshRefs = useRef<(THREE.Mesh | null)[]>([]);
    const progressRef = useRef<number[]>([]);

    const trails = useMemo<ShootingStar[]>(() => {
        const arr: ShootingStar[] = [];
        for (let i = 0; i < 6; i++) {
            // Random positions in the far background
            const startX = (Math.random() - 0.5) * 80;
            const startY = 10 + Math.random() * 30;
            const startZ = -20 - Math.random() * 40;

            // Direction — mostly horizontal with slight downward
            const dirX = (Math.random() - 0.5) * 40;
            const dirY = -5 - Math.random() * 10;
            const dirZ = (Math.random() - 0.5) * 20;

            arr.push({
                startPos: new THREE.Vector3(startX, startY, startZ),
                endPos: new THREE.Vector3(startX + dirX, startY + dirY, startZ + dirZ),
                speed: 0.15 + Math.random() * 0.2,
                delay: Math.random() * 15,
                length: 1.5 + Math.random() * 2.5,
                brightness: 0.3 + Math.random() * 0.5,
            });
        }
        return arr;
    }, []);

    // Initialize progress
    if (progressRef.current.length === 0) {
        progressRef.current = trails.map(t => -t.delay);
    }

    useFrame((_, delta) => {
        trails.forEach((trail, i) => {
            progressRef.current[i] += delta * trail.speed;
            const mesh = meshRefs.current[i];
            if (!mesh) return;

            const p = progressRef.current[i];

            if (p < 0 || p > 1) {
                mesh.visible = false;
                if (p > 1) {
                    // Reset with new random delay
                    progressRef.current[i] = -(Math.random() * 12 + 5);
                    // Randomize position for next pass
                    trail.startPos.set(
                        (Math.random() - 0.5) * 80,
                        10 + Math.random() * 30,
                        -20 - Math.random() * 40
                    );
                    const dirX = (Math.random() - 0.5) * 40;
                    const dirY = -5 - Math.random() * 10;
                    const dirZ = (Math.random() - 0.5) * 20;
                    trail.endPos.copy(trail.startPos).add(new THREE.Vector3(dirX, dirY, dirZ));
                }
                return;
            }

            mesh.visible = true;
            // Position along path
            const pos = trail.startPos.clone().lerp(trail.endPos, p);
            mesh.position.copy(pos);

            // Orient along travel direction
            const dir = trail.endPos.clone().sub(trail.startPos).normalize();
            mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);

            // Fade in/out
            const fade = p < 0.15 ? p / 0.15 : p > 0.85 ? (1 - p) / 0.15 : 1;
            const mat = mesh.material as THREE.MeshBasicMaterial;
            mat.opacity = fade * trail.brightness;
        });
    });

    return (
        <group ref={trailsRef}>
            {trails.map((trail, i) => (
                <mesh
                    key={i}
                    ref={(el) => { meshRefs.current[i] = el; }}
                    visible={false}
                >
                    <cylinderGeometry args={[0.02, 0.005, trail.length, 4]} />
                    <meshBasicMaterial
                        color="#ffffff"
                        transparent
                        opacity={0}
                        depthWrite={false}
                        blending={THREE.AdditiveBlending}
                    />
                </mesh>
            ))}
        </group>
    );
}
