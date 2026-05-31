import { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { Stars, Preload } from '@react-three/drei';
import Earth from './Earth';
import Satellite from './Satellite';
import OrbitTrail from './OrbitTrail';
import LightTrails from './LightTrails';
import SunRays from './SunRays';

export default function EarthScene() {
    return (
        <div className="fixed inset-0 z-[1]" style={{ pointerEvents: 'none' }}>
            <Canvas
                camera={{ position: [-3, 3, 16], fov: 40 }}
                gl={{
                    antialias: true,
                    alpha: true,
                    powerPreference: 'high-performance',
                    stencil: false,
                    depth: true,
                }}
                style={{ background: 'transparent' }}
                dpr={[1, 1.5]}
                performance={{ min: 0.5 }}
            >
                {/* Lighting */}
                <directionalLight position={[8, 4, 6]} intensity={2.0} color="#fff8f0" />
                <ambientLight intensity={0.08} color="#4466aa" />
                <pointLight position={[-6, -4, -8]} intensity={0.4} color="#7EC8E3" distance={30} />

                {/* Starfield */}
                <Stars radius={120} depth={60} count={2000} factor={3.5} saturation={0.1} fade speed={0.3} />

                {/* Light trails */}
                <LightTrails />

                {/* Volumetric sun rays */}
                <SunRays />

                <Suspense fallback={null}>
                    <group position={[-2, -2, 0]}>
                        <Earth />
                        <Satellite />
                        <OrbitTrail />
                    </group>
                    <Preload all />
                </Suspense>
            </Canvas>
        </div>
    );
}
