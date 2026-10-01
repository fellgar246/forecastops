import { afterEach, expect, test, vi } from "vitest";
import { listDatasets } from "@/lib/api/client";
import { authEnabled, clearAccessToken, readAccessToken, signIn, storeAccessToken } from "@/lib/auth";

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  clearAccessToken();
});

test("local mode leaves sign-in disabled", () => {
  vi.stubEnv("NEXT_PUBLIC_AUTH_ENABLED", "false");
  expect(authEnabled()).toBe(false);
});

test("cloud auth stores an access token and the client sends it", async () => {
  vi.stubEnv("NEXT_PUBLIC_AUTH_ENABLED", "true");
  vi.stubEnv("NEXT_PUBLIC_COGNITO_CLIENT_ID", "web-client");
  vi.stubEnv("NEXT_PUBLIC_COGNITO_REGION", "us-east-1");
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.includes("cognito-idp")) {
      const body = JSON.parse(String(init?.body));
      expect(body.AuthParameters.USERNAME).toBe("ada");
      expect(body.ClientId).toBe("web-client");
      return new Response(JSON.stringify({ AuthenticationResult: { AccessToken: "access-token" } }), {
        status: 200,
      });
    }
    return new Response(JSON.stringify({ items: [] }), { status: 200 });
  });
  vi.stubGlobal("fetch", fetchMock);

  await signIn("ada", "secret");
  expect(readAccessToken()).toBe("access-token");
  await listDatasets();
  const datasetCall = fetchMock.mock.calls.find((call) => String(call[0]).endsWith("/datasets"));
  const headers = new Headers(datasetCall?.[1]?.headers);
  expect(headers.get("Authorization")).toBe("Bearer access-token");
});

test("local mode does not attach a stored token", async () => {
  vi.stubEnv("NEXT_PUBLIC_AUTH_ENABLED", "false");
  storeAccessToken("leftover");
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    expect(String(input)).toContain("/datasets");
    const headers = new Headers(init?.headers);
    expect(headers.get("Authorization")).toBeNull();
    return new Response(JSON.stringify({ items: [] }), { status: 200 });
  });
  vi.stubGlobal("fetch", fetchMock);

  await listDatasets();
  expect(fetchMock).toHaveBeenCalledOnce();
});
