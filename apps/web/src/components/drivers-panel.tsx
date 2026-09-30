import type { Explanation } from "@/lib/api/schemas";

export function DriversPanel({
  family,
  signals = [],
}: {
  family: string;
  signals?: Explanation["signals"];
}) {
  const tree = family === "gradient_boosting";
  const up = signals.filter((signal) => signal.direction === "positive");
  const down = signals.filter((signal) => signal.direction === "negative");
  const context = signals.filter((signal) => signal.direction == null);
  return (
    <section className="card">
      <h2>{tree ? "Top signals for this forecast" : "Context for this forecast"}</h2>
      {tree ? null : (
        <p className="caption">These are signals the model had available, not proven causes.</p>
      )}
      {signals.length === 0 ? <p>This forecast package does not include signal contributions.</p> : null}
      {tree && up.length > 0 ? (
        <>
          <h3>Pushing demand up</h3>
          <ul>
            {up.map((signal) => (
              <li key={signal.name}>{signal.name}</li>
            ))}
          </ul>
        </>
      ) : null}
      {tree && down.length > 0 ? (
        <>
          <h3>Pushing demand down</h3>
          <ul>
            {down.map((signal) => (
              <li key={signal.name}>{signal.name}</li>
            ))}
          </ul>
        </>
      ) : null}
      {!tree && context.length > 0 ? (
        <ul>
          {context.map((signal) => (
            <li key={signal.name}>{signal.name.replaceAll("_", " ")}</li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}
