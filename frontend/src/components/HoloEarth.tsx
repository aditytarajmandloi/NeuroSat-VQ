import { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Stars } from '@react-three/drei';
import * as THREE from 'three';

function Earth() {
    const earthRef = useRef<THREE.Mesh>(null);
    const orbitGroupRef = useRef<THREE.Group>(null);

    useFrame((_, delta) => {
        if (orbitGroupRef.current) {
            // Satellite orbit speed and axis
            orbitGroupRef.current.rotation.y += delta * 0.4;
            orbitGroupRef.current.rotation.z = Math.sin(orbitGroupRef.current.rotation.y * 0.5) * 0.2;
        }
    });

    return (
        <group>
            {/* The Core Earth */}
            <mesh ref={earthRef}>
                <sphereGeometry args={[3.8, 64, 64]} />
                {/* Solid black core to hide stars behind it */}
                <meshBasicMaterial color="#000000" />

                {/* The Cyan Holographic Grid Wireframe */}
                <mesh>
                    <sphereGeometry args={[3.82, 32, 32]} />
                    <meshBasicMaterial color="#66FCF1" wireframe transparent opacity={0.4} />
                </mesh>
            </mesh>

            {/* Satellite Orbit Structure */}
            <group ref={orbitGroupRef}>
                {/* Satellite Object Positioned away from center */}
                <group position={[5.2, 0, 0]}>
                    {/* Main Satellite Chassis */}
                    <mesh>
                        <boxGeometry args={[0.5, 0.5, 1.0]} />
                        <meshBasicMaterial color="#7B2CBF" wireframe />
                    </mesh>

                    {/* Solar Panel Array */}
                    <mesh position={[0, 0.75, 0]}>
                        <boxGeometry args={[2.0, 0.25, 0.75]} />
                        <meshBasicMaterial color="#66FCF1" wireframe transparent opacity={0.4} />
                    </mesh>

                    {/* Core Pulse Glow */}
                    <mesh>
                        <sphereGeometry args={[0.2, 16, 16]} />
                        <meshBasicMaterial color="#66FCF1" />
                    </mesh>
                </group>
            </group>
        </group>
    );
}

export default function HoloEarth() {
    return (
        <div className="fixed top-0 left-0 w-[100vw] h-[100vh] z-[-2] bg-black">
            <Canvas camera={{ position: [0, 1.5, 14], fov: 45 }}>
                <color attach="background" args={['#000000']} />

                {/* Deep Space Starfield */}
                <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />

                <ambientLight intensity={0.5} />

                <Earth />

                {/* Allows the user to slowly pan around the scene natively */}
                <OrbitControls
                    enableZoom={false}
                    enablePan={false}
                    maxPolarAngle={Math.PI / 1.5}
                    minPolarAngle={Math.PI / 3}
                />
            </Canvas>
        </div>
    );
}
