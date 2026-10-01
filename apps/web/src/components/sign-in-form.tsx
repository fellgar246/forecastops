"use client";

import { useState, type FormEvent } from "react";
import { signIn } from "@/lib/auth";

export function SignInForm({ onSignedIn }: { onSignedIn: () => void }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setPending(true);
    setError("");
    try {
      await signIn(username, password);
      onSignedIn();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Sign-in failed. Check the username and password.");
    } finally {
      setPending(false);
    }
  }

  return (
    <main className="sign-in">
      <form className="sign-in-card" onSubmit={submit}>
        <h1>Sign in</h1>
        <p className="lede">Use the account for your organization.</p>
        <label className="field">
          Username
          <input
            name="username"
            autoComplete="username"
            value={username}
            onChange={(event) => setUsername(event.target.value)}
            required
          />
        </label>
        <label className="field">
          Password
          <input
            name="password"
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            required
          />
        </label>
        {error ? (
          <p role="alert" className="sign-in-error">
            {error}
          </p>
        ) : null}
        <button className="btn btn-primary" type="submit" disabled={pending}>
          {pending ? "Signing in" : "Sign in"}
        </button>
      </form>
    </main>
  );
}
