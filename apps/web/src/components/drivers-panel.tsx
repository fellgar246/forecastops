export function DriversPanel({ family }: { family: string }) {
  const tree = family === "gradient_boosting";
  return (
    <section className="card">
      <h2>{tree ? "Top signals for this forecast" : "Context for this forecast"}</h2>
      {tree ? null : (
        <p className="caption">These are signals the model had available, not proven causes.</p>
      )}
      <p>This forecast package does not include signal contributions.</p>
    </section>
  );
}
