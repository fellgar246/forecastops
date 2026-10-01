const TOKEN_KEY = "forecastops.access_token";

function publicEnv(name: "NEXT_PUBLIC_COGNITO_CLIENT_ID" | "NEXT_PUBLIC_COGNITO_REGION"): string {
  const value =
    name === "NEXT_PUBLIC_COGNITO_CLIENT_ID"
      ? process.env.NEXT_PUBLIC_COGNITO_CLIENT_ID
      : process.env.NEXT_PUBLIC_COGNITO_REGION;
  return typeof value === "string" ? value : "";
}

export function authEnabled(): boolean {
  return process.env.NEXT_PUBLIC_AUTH_ENABLED === "true";
}

export function readAccessToken(): string | null {
  if (typeof window === "undefined") {
    return null;
  }
  const token = window.sessionStorage.getItem(TOKEN_KEY);
  return token && token.length > 0 ? token : null;
}

export function storeAccessToken(token: string): void {
  window.sessionStorage.setItem(TOKEN_KEY, token);
}

export function clearAccessToken(): void {
  window.sessionStorage.removeItem(TOKEN_KEY);
}

export async function signIn(username: string, password: string): Promise<string> {
  const clientId = publicEnv("NEXT_PUBLIC_COGNITO_CLIENT_ID");
  if (!clientId) {
    throw new Error("Sign-in is not configured.");
  }
  const region = publicEnv("NEXT_PUBLIC_COGNITO_REGION") || "us-east-1";
  let response: Response;
  try {
    response = await fetch(`https://cognito-idp.${region}.amazonaws.com/`, {
      method: "POST",
      headers: {
        "content-type": "application/x-amz-json-1.1",
        "x-amz-target": "AWSCognitoIdentityProviderService.InitiateAuth",
      },
      body: JSON.stringify({
        AuthFlow: "USER_PASSWORD_AUTH",
        ClientId: clientId,
        AuthParameters: { USERNAME: username, PASSWORD: password },
      }),
    });
  } catch {
    throw new Error("Sign-in failed. Check the username and password.");
  }
  const payload: unknown = await response.json().catch(() => null);
  const token = accessToken(payload);
  if (!response.ok || !token) {
    throw new Error("Sign-in failed. Check the username and password.");
  }
  storeAccessToken(token);
  return token;
}

function accessToken(payload: unknown): string | null {
  if (!payload || typeof payload !== "object") {
    return null;
  }
  const result = (payload as { AuthenticationResult?: { AccessToken?: unknown } }).AuthenticationResult;
  const token = result?.AccessToken;
  return typeof token === "string" && token.length > 0 ? token : null;
}
