"use client";

import { Float, Line, PerspectiveCamera, Sparkles } from "@react-three/drei";
import { Canvas, useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";

type PointerState = {
  x: number;
  y: number;
};

function round(value: number, precision = 3) {
  return Number(value.toFixed(precision));
}

function buildRoseCurve(
  petals: number,
  radius: number,
  steps: number,
  rotation: number,
  yScale = 1,
  z = 0,
) {
  const points: [number, number, number][] = [];

  for (let index = 0; index <= steps; index += 1) {
    const theta = (index / steps) * Math.PI * 2;
    const mod =
      Math.sin(theta * petals) * radius * (0.88 + Math.sin(theta * 3.2) * 0.12);
    points.push([
      round(Math.cos(theta + rotation) * mod),
      round(Math.sin(theta + rotation) * mod * yScale),
      z,
    ]);
  }

  return points;
}

function buildSpiral(
  turns: number,
  scale: number,
  rotation: number,
  z: number,
) {
  const points: [number, number, number][] = [];

  for (let theta = 0.24; theta <= turns * Math.PI * 2; theta += 0.16) {
    const radius = 0.18 + theta * scale;
    points.push([
      round(Math.cos(theta + rotation) * radius),
      round(Math.sin(theta + rotation) * radius),
      z,
    ]);
  }

  return points;
}

function buildRoute(points: [number, number, number][]) {
  return points.map(
    ([x, y, z]) => [round(x), round(y), round(z)] as [number, number, number],
  );
}

function buildNodes(
  count: number,
  radius: number,
  rotation: number,
  z: number,
) {
  return Array.from({ length: count }, (_, index) => {
    const angle = rotation + (index / count) * Math.PI * 2;
    return [
      round(Math.cos(angle) * radius),
      round(Math.sin(angle) * radius * 0.84),
      z,
    ] as [number, number, number];
  });
}

function BlueprintField({
  pointerRef,
}: {
  pointerRef: React.MutableRefObject<PointerState>;
}) {
  const stageRef = useRef<THREE.Group>(null);
  const orbitRef = useRef<THREE.Group>(null);
  const roseA = useMemo(
    () => buildRoseCurve(13, 2.95, 560, 0.12, 1.02, -1),
    [],
  );
  const roseB = useMemo(
    () => buildRoseCurve(8, 3.2, 420, 0.48, 1.08, -1.2),
    [],
  );
  const roseC = useMemo(
    () => buildRoseCurve(21, 2.56, 640, -0.3, 0.96, -0.86),
    [],
  );
  const spiralA = useMemo(() => buildSpiral(5.4, 0.114, 0, -0.72), []);
  const spiralB = useMemo(() => buildSpiral(5.4, 0.114, Math.PI, -0.74), []);
  const spiralC = useMemo(() => buildSpiral(4.2, 0.09, Math.PI / 2, -0.76), []);
  const routeA = useMemo(
    () =>
      buildRoute([
        [-4.7, 1.56, -1.34],
        [-3.1, 1.56, -1.34],
        [-2.1, 0.74, -1.34],
        [-0.8, 0.74, -1.34],
        [0.6, 1.86, -1.34],
        [3.8, 1.86, -1.34],
      ]),
    [],
  );
  const routeB = useMemo(
    () =>
      buildRoute([
        [-4.4, -0.72, -1.44],
        [-2.6, -0.72, -1.44],
        [-1.2, -1.64, -1.44],
        [0.7, -1.64, -1.44],
        [2.1, -0.86, -1.44],
        [4.2, -0.86, -1.44],
      ]),
    [],
  );
  const routeC = useMemo(
    () =>
      buildRoute([
        [-3.34, 3.0, -1.56],
        [-3.34, 1.94, -1.56],
        [-1.5, 1.94, -1.56],
        [-1.5, 3.14, -1.56],
      ]),
    [],
  );
  const orbitNodes = useMemo(() => buildNodes(10, 3.06, 0.2, -1.02), []);

  useFrame((state) => {
    const stepped = Math.floor(state.clock.elapsedTime * 8) / 8;
    const { x, y } = pointerRef.current;

    if (stageRef.current) {
      const targetZ = Math.sin(stepped * 0.62) * 0.08 + x * 0.22;
      const targetX = Math.cos(stepped * 0.46) * 0.04 - y * 0.14;
      const targetY = Math.sin(stepped * 0.88) * 0.07;

      stageRef.current.rotation.z = THREE.MathUtils.lerp(
        stageRef.current.rotation.z,
        targetZ,
        0.08,
      );
      stageRef.current.rotation.x = THREE.MathUtils.lerp(
        stageRef.current.rotation.x,
        targetX,
        0.08,
      );
      stageRef.current.position.y = THREE.MathUtils.lerp(
        stageRef.current.position.y,
        targetY,
        0.08,
      );
    }

    if (orbitRef.current) {
      orbitRef.current.rotation.z = stepped * 0.16;
      orbitRef.current.rotation.y = x * 0.12;
    }
  });

  return (
    <>
      <PerspectiveCamera fov={35} makeDefault position={[0, 0, 10.5]} />
      <ambientLight intensity={0.94} />
      <pointLight color="#f7d86f" intensity={22} position={[0.4, 1.2, 4]} />
      <pointLight color="#1948ff" intensity={11} position={[-5.2, 1.8, 3]} />
      <pointLight color="#ffffff" intensity={6} position={[5.2, -1.4, 3]} />

      <Float floatIntensity={0.12} rotationIntensity={0.04} speed={0.9}>
        <group position={[0, 0, 0]} scale={[1.34, 1.34, 1.34]}>
          <group ref={stageRef} position={[0, 0, 0]}>
            <mesh position={[0, 0, -2.4]}>
              <circleGeometry args={[1.58, 72]} />
              <meshBasicMaterial color="#f0c942" opacity={0.18} transparent />
            </mesh>

            <mesh position={[0, 0, -2.6]} rotation={[Math.PI / 2, 0, 0]}>
              <torusGeometry args={[1.82, 0.026, 16, 240]} />
              <meshBasicMaterial color="#f0c942" opacity={0.16} transparent />
            </mesh>

            <mesh position={[0, 0, -2.8]} rotation={[Math.PI / 2, 0, 0.22]}>
              <torusGeometry args={[2.2, 0.018, 12, 240]} />
              <meshBasicMaterial color="#0019a8" opacity={0.13} transparent />
            </mesh>

            <Line
              color="#0d3fe1"
              lineWidth={1.4}
              opacity={0.34}
              points={roseA}
              transparent
            />
            <Line
              color="#295cff"
              lineWidth={1.15}
              opacity={0.28}
              points={roseB}
              transparent
            />
            <Line
              color="#1442c9"
              lineWidth={1.05}
              opacity={0.28}
              points={roseC}
              transparent
            />

            <Line
              color="#f0c942"
              lineWidth={1.2}
              opacity={0.42}
              points={spiralA}
              transparent
            />
            <Line
              color="#fff4c9"
              lineWidth={0.95}
              opacity={0.3}
              points={spiralB}
              transparent
            />
            <Line
              color="#f0c942"
              lineWidth={0.7}
              opacity={0.18}
              points={spiralC}
              transparent
            />

            <Line
              color="#0f3edb"
              lineWidth={2.2}
              opacity={0.22}
              points={routeA}
              transparent
            />
            <Line
              color="#ffffff"
              lineWidth={1.5}
              opacity={0.18}
              points={routeB}
              transparent
            />
            <Line
              color="#f0c942"
              lineWidth={1.8}
              opacity={0.24}
              points={routeC}
              transparent
            />

            <group ref={orbitRef}>
              {Array.from({ length: 28 }, (_, index) => {
                const angle = (index / 28) * Math.PI * 2;
                const radius = 2.7 + Math.sin(index * 1.22) * 0.08;
                return (
                  <mesh
                    key={`tick-${index}`}
                    position={[
                      round(Math.cos(angle) * radius),
                      round(Math.sin(angle) * radius * 0.82),
                      -0.94,
                    ]}
                    rotation={[0, 0, angle]}
                  >
                    <planeGeometry
                      args={[
                        index % 4 === 0 ? 0.2 : 0.11,
                        index % 3 === 0 ? 1.1 : 0.72,
                      ]}
                    />
                    <meshBasicMaterial
                      color={index % 2 === 0 ? "#f0c942" : "#e8efff"}
                      opacity={index % 4 === 0 ? 0.22 : 0.12}
                      transparent
                    />
                  </mesh>
                );
              })}
            </group>

            {orbitNodes.map(([x, y, z], index) => (
              <group key={`node-${index}`} position={[x, y, z]}>
                <mesh>
                  <circleGeometry args={[0.15, 28]} />
                  <meshBasicMaterial color="#0019a8" />
                </mesh>
                <mesh position={[0, 0, 0.01]}>
                  <circleGeometry args={[0.085, 28]} />
                  <meshBasicMaterial color="#ffffff" />
                </mesh>
                <mesh position={[0, 0, 0.02]}>
                  <circleGeometry args={[0.03, 18]} />
                  <meshBasicMaterial color="#f0c942" />
                </mesh>
              </group>
            ))}
          </group>
        </group>
      </Float>

      <Sparkles
        color="#8fb1ff"
        count={92}
        opacity={0.22}
        scale={[13.5, 10, 2]}
        size={2.4}
        speed={0.08}
      />
      <Sparkles
        color="#f0c942"
        count={46}
        opacity={0.16}
        scale={[9, 7, 2]}
        size={3}
        speed={0.04}
      />
    </>
  );
}

export default function SunflowerHero() {
  const pointerRef = useRef<PointerState>({ x: 0, y: 0 });

  const handleMove = (event: React.MouseEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const px = (event.clientX - rect.left) / rect.width;
    const py = (event.clientY - rect.top) / rect.height;

    pointerRef.current = {
      x: round((px - 0.5) * 2, 2),
      y: round((py - 0.5) * 2, 2),
    };
  };

  const handleLeave = () => {
    pointerRef.current = { x: 0, y: 0 };
  };

  return (
    <div
      className="absolute inset-0"
      onMouseLeave={handleLeave}
      onMouseMove={handleMove}
    >
      <div
        className="absolute inset-0"
        style={{
          background:
            "radial-gradient(circle at 50% 50%, rgba(240,201,66,0.10), transparent 18%), radial-gradient(circle at 50% 50%, rgba(0,25,168,0.08), transparent 44%), linear-gradient(180deg, #fbfcfd 0%, #f5f7f9 100%)",
        }}
      />

      <div
        className="absolute inset-0 opacity-[0.18]"
        style={{
          backgroundImage:
            "linear-gradient(rgba(0,25,168,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(0,25,168,0.05) 1px, transparent 1px)",
          backgroundSize: "52px 52px",
        }}
      />

      <div className="pointer-events-none absolute left-[calc(50%+40px)] top-1/2 h-[122vh] w-[122vw] min-h-[860px] min-w-[860px] -translate-x-1/2 -translate-y-1/2 sm:left-[calc(50%+52px)] sm:h-[128vh] sm:w-[128vw] lg:left-[calc(50%+64px)] lg:h-[138vh] lg:w-[138vw]">
        <Canvas dpr={[1, 2]} gl={{ alpha: true, antialias: true }}>
          <BlueprintField pointerRef={pointerRef} />
        </Canvas>
      </div>
    </div>
  );
}
