import { useMemo } from 'react';
import { Line } from '@react-three/drei';

interface OrbitTrailProps {
    orbits?: { radius: number; inclination: number; color: string }[];
}

const DEFAULT_ORBITS = [
    { radius: 7, inclination: 0.26, color: '#E8A838' },
    { radius: 8.5, inclination: -0.52, color: '#7EC8E3' },
    { radius: 6.2, inclination: 0.87, color: '#bb88ee' },
];

export default function OrbitTrail({ orbits = DEFAULT_ORBITS }: OrbitTrailProps) {
    return (
        <group>
            {orbits.map((orbit, idx) => (
                <OrbitRing key={idx} radius={orbit.radius} inclination={orbit.inclination} color={orbit.color} />
            ))}
        </group>
    );
}

function OrbitRing({ radius, inclination, color }: { radius: number; inclination: number; color: string }) {
    const points = useMemo(() => {
        const pts: [number, number, number][] = [];
        const segs = 96;
        for (let i = 0; i <= segs; i++) {
            const angle = (i / segs) * Math.PI * 2;
            pts.push([Math.cos(angle) * radius, 0, Math.sin(angle) * radius]);
        }
        return pts;
    }, [radius]);

    return (
        <group rotation={[inclination, 0, 0]}>
            <Line
                points={points}
                color={color}
                transparent
                opacity={0.04}
                lineWidth={0.5}
                depthWrite={false}
            />
        </group>
    );
}
