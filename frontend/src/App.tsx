import { useState, useRef, useLayoutEffect } from "react";
import { Rnd } from "react-rnd";

export default function App() {
  const [obstacles, setObstacles] = useState<any[]>([]);
  const [start, setStart] = useState({ x: 0.1, y: 0.1 });
  const [goal, setGoal] = useState({ x: 0.9, y: 0.9 });
  const [imageUrl, setImageUrl] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 400 });

  useLayoutEffect(() => {
    const resize = () => {
      if (containerRef.current) {
        const { offsetWidth, offsetHeight } = containerRef.current;
        console.log("Measured size:", offsetWidth, offsetHeight);
        setDimensions({ width: offsetWidth, height: offsetHeight });
      }
    };
    window.addEventListener("resize", resize);
    resize();
    return () => window.removeEventListener("resize", resize);
  }, []);

  const handleDropObstacle = (type: string, x: number, y: number) => {
    const newObstacle = {
      id: crypto.randomUUID(),
      type,
      center: [x, y],
      radius: type === "circle" ? 0.1 : undefined,
      size: type === "square" ? 0.1 : undefined,
      margin: 0.05,
    };
    setObstacles((prev) => [...prev, newObstacle]);
  };

  const generateYaml = () => {
    const body = {
      shape: "dot",
      dynamic: "point_2nd",
      goal_mode: "center",
      start_state: [start.x, start.y, 0.0, 0.0],
      goal_state: [goal.x, goal.y, 0.0, 0.0],
      control_bounds: [-1.0, 1.0],
    };

    const solver = {
      N: 40,
      dt: 0.1,
      mode: "casadi",
    };

    const yaml = { body, obstacles, solver };

    fetch("http://localhost:8000/generate-and-run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config: yaml }),
    })
      .then((res) => res.json())
      .then((data) => {
        if (data.image_path) {
          setImageUrl("http://localhost:8000/" + data.image_path + "?" + Date.now());
        }
      })
      .catch((err) => alert("Error: " + err));
  };

  const updateObstacle = (id: string, center: [number, number], size?: number) => {
    setObstacles((prev) =>
      prev.map((o) =>
        o.id === id ? { ...o, center, radius: o.radius ?? size, size: o.size ?? size } : o
      )
    );
  };

  const deleteSelected = () => {
    if (selectedId) setObstacles((prev) => prev.filter((o) => o.id !== selectedId));
    setSelectedId(null);
  };

  const renderRndObstacle = (o: any) => {
    const size = o.radius ?? o.size;
    const width = size * dimensions.width;
    const height = size * dimensions.height;
    const x = (o.center[0] - size / 2) * dimensions.width;
    const y = (1 - o.center[1] - size / 2) * dimensions.height;

    return (
      <Rnd
        key={o.id}
        size={{ width, height }}
        position={{ x, y }}
        onDragStop={(e, d) => {
          const newX = d.x / dimensions.width + size / 2;
          const newY = 1 - (d.y / dimensions.height + size / 2);
          updateObstacle(o.id, [newX, newY]);
        }}
        onResizeStop={(e, direction, ref, delta, position) => {
          const newSize = ref.offsetWidth / dimensions.width;
          const newX = position.x / dimensions.width + newSize / 2;
          const newY = 1 - (position.y / dimensions.height + newSize / 2);
          updateObstacle(o.id, [newX, newY], newSize);
        }}
        bounds="parent"
        onClick={() => setSelectedId(o.id)}
        style={{
          backgroundColor: o.id === selectedId ? "blue" : "black",
          borderRadius: o.type === "circle" ? "50%" : "0",
          transform: "scaleY(-1)",
        }}
      />
    );
  };

  return (
    <div style={{ padding: "1rem" }}>
      <h1>Yaml Generator</h1>
      <div
        ref={containerRef}
        style={{
          border: "1px solid black",
          height: "400px",
          width: "800px",
          position: "relative",
          transform: "scaleY(-1)",
        }}
      >
        <div
          style={{
            position: "absolute",
            left: `${start.x * 100}%`,
            top: `${(1 - start.y) * 100}%`,
            transform: "scaleY(-1)",
            width: "10px",
            height: "10px",
            backgroundColor: "green",
            borderRadius: "50%",
            cursor: "pointer",
          }}
          draggable
          onDragEnd={(e) => {
            const rect = (e.currentTarget.parentElement as HTMLElement).getBoundingClientRect();
            setStart({
              x: (e.clientX - rect.left) / rect.width,
              y: 1 - (e.clientY - rect.top) / rect.height,
            });
          }}
        ></div>

        <div
          style={{
            position: "absolute",
            left: `${goal.x * 100}%`,
            top: `${(1 - goal.y) * 100}%`,
            transform: "scaleY(-1)",
            width: "10px",
            height: "10px",
            backgroundColor: "red",
            borderRadius: "50%",
            cursor: "pointer",
          }}
          draggable
          onDragEnd={(e) => {
            const rect = (e.currentTarget.parentElement as HTMLElement).getBoundingClientRect();
            setGoal({
              x: (e.clientX - rect.left) / rect.width,
              y: 1 - (e.clientY - rect.top) / rect.height,
            });
          }}
        ></div>

        {obstacles.map(renderRndObstacle)}
      </div>

      <div style={{ marginTop: "1rem", display: "flex", gap: "0.5rem" }}>
        <button onClick={() => handleDropObstacle("square", 0.5, 0.5)}>Add Square</button>
        <button onClick={() => handleDropObstacle("circle", 0.7, 0.3)}>Add Circle</button>
        <button onClick={generateYaml}>Generate YAML & Run</button>
        <button onClick={deleteSelected}>Delete Selected</button>
      </div>

      {imageUrl && (
        <div style={{ marginTop: "1rem" }}>
          <h3>Result Image:</h3>
          <img src={imageUrl} alt="Path result" style={{ maxWidth: "100%", border: "1px solid #ccc" }} />
        </div>
      )}
    </div>
  );
}
