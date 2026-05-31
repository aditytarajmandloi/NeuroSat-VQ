import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { useTexture } from '@react-three/drei';
import * as THREE from 'three';

// Custom Atmosphere Shader
const atmosphereVertexShader = `
  varying vec3 vNormal;
  varying vec3 vPosition;
  void main() {
    vNormal = normalize(normalMatrix * normal);
    vPosition = (modelViewMatrix * vec4(position, 1.0)).xyz;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const atmosphereFragmentShader = `
  varying vec3 vNormal;
  varying vec3 vPosition;
  uniform vec3 glowColor;
  uniform float intensity;
  uniform float power;
  void main() {
    vec3 viewDir = normalize(-vPosition);
    float fresnel = pow(1.0 - max(dot(viewDir, vNormal), 0.0), power);
    gl_FragColor = vec4(glowColor, fresnel * intensity);
  }
`;

export default function Earth() {
    const earthRef = useRef<THREE.Group>(null);
    const cloudsRef = useRef<THREE.Mesh>(null);
    const nightRef = useRef<THREE.Mesh>(null);

    // Load NASA textures
    const [dayMap, nightMap, bumpMap, specularMap] = useTexture([
        '/textures/earth-blue-marble.jpg',
        '/textures/earth-night.jpg',
        '/textures/earth-topology.png',
        '/textures/earth-water.png',
    ]);

    // Atmosphere shader uniforms
    const atmosphereUniforms = useMemo(() => ({
        glowColor: { value: new THREE.Color(0.3, 0.6, 1.0) },
        intensity: { value: 0.35 },
        power: { value: 4.0 },
    }), []);

    useFrame((_, delta) => {
        if (earthRef.current) {
            // Slow Earth rotation
            earthRef.current.rotation.y += delta * 0.025;
        }
        if (cloudsRef.current) {
            // Clouds rotate slightly faster than Earth
            cloudsRef.current.rotation.y += delta * 0.035;
        }
    });

    const EARTH_RADIUS = 4;

    return (
        <group ref={earthRef} rotation={[0.408, 0, 0]}> {/* 23.4 degree axial tilt */}
            {/* Main Earth — Day Side */}
            <mesh>
                <sphereGeometry args={[EARTH_RADIUS, 128, 128]} />
                <meshStandardMaterial
                    map={dayMap}
                    bumpMap={bumpMap}
                    bumpScale={0.04}
                    metalness={0.1}
                    roughness={0.7}
                    metalnessMap={specularMap}
                />
            </mesh>

            {/* Night Side — City Lights (Additive blend on top) */}
            <mesh ref={nightRef}>
                <sphereGeometry args={[EARTH_RADIUS + 0.005, 128, 128]} />
                <meshBasicMaterial
                    map={nightMap}
                    transparent
                    opacity={0.85}
                    blending={THREE.AdditiveBlending}
                    depthWrite={false}
                />
            </mesh>

            {/* Cloud Layer */}
            <mesh ref={cloudsRef}>
                <sphereGeometry args={[EARTH_RADIUS + 0.04, 64, 64]} />
                <meshStandardMaterial
                    color="#ffffff"
                    transparent
                    opacity={0.12}
                    roughness={1}
                    metalness={0}
                    depthWrite={false}
                />
            </mesh>

            {/* Atmosphere Glow — Fresnel Shader */}
            <mesh>
                <sphereGeometry args={[EARTH_RADIUS + 0.2, 64, 64]} />
                <shaderMaterial
                    vertexShader={atmosphereVertexShader}
                    fragmentShader={atmosphereFragmentShader}
                    uniforms={atmosphereUniforms}
                    transparent
                    side={THREE.BackSide}
                    depthWrite={false}
                    blending={THREE.AdditiveBlending}
                />
            </mesh>

            {/* Inner atmosphere rim (visible from front) */}
            <mesh>
                <sphereGeometry args={[EARTH_RADIUS + 0.08, 64, 64]} />
                <shaderMaterial
                    vertexShader={atmosphereVertexShader}
                    fragmentShader={atmosphereFragmentShader}
                    uniforms={useMemo(() => ({
                        glowColor: { value: new THREE.Color(0.4, 0.7, 1.0) },
                        intensity: { value: 0.1 },
                        power: { value: 6.0 },
                    }), [])}
                    transparent
                    side={THREE.FrontSide}
                    depthWrite={false}
                    blending={THREE.AdditiveBlending}
                />
            </mesh>
        </group>
    );
}
