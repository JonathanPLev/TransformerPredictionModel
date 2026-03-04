(function () {
    "use strict";

    const API_BASE = (window.__PREDICT_API_BASE__ || "").replace(/\/+$/, "");
    const PREDICT_ENDPOINT = API_BASE + "/predict";
    const REQUEST_TIMEOUT_MS = 30_000;

    const MAX_NAME_LENGTH = 64;
    const ALLOWED_CHARS = /^[a-zA-Z\s\-']+$/;
    const CONTROL_CHARS = /[\x00-\x1F\x7F-\x9F]/g;
    const FANCY_APOSTROPHES = /[\u2018\u2019\u201A\u201B\u0060\u00B4]/g;

    function sanitizePlayerName(raw) {
        if (typeof raw !== "string") return null;

        let s = raw.trim();
        if (s.length === 0) return null;

        s = s.replace(CONTROL_CHARS, "");
        s = s.replace(/\s+/g, " ").trim();
        s = s.replace(FANCY_APOSTROPHES, "'");

        if (s.length > MAX_NAME_LENGTH) {
            s = s.substring(0, MAX_NAME_LENGTH).trim();
        }

        if (!ALLOWED_CHARS.test(s)) return null;
        if (!/[a-zA-Z]/.test(s)) return null;

        return s;
    }

    async function sendPredictRequest(cleanedName) {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

        try {
            const res = await fetch(PREDICT_ENDPOINT, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                body: JSON.stringify({ query: cleanedName }),
                signal: controller.signal,
            });
            return res;
        } finally {
            clearTimeout(timeout);
        }
    }

    function decodeResponse(httpStatus, body) {
        const result = {
            statusCode: httpStatus,
            playerResult: null,
            officialPlayerName: null,
            teamAgainst: null,
            timeAndDateEST: null,
            errorMessage: null,
        };

        if (!body || typeof body !== "object") {
            result.errorMessage = "Received an invalid response from the server.";
            return result;
        }

        if (typeof body.prediction === "number") {
            result.playerResult = body.prediction;
        }

        const meta = body.metadata || {};
        result.officialPlayerName =
            meta.player_name || meta.cleaned_query || null;
        result.teamAgainst =
            meta.opponent || meta.team_against || null;
        result.timeAndDateEST =
            meta.game_time || meta.game_datetime_est || null;

        const errorType = meta.error_type || null;

        if (httpStatus === 404) {
            result.errorMessage =
                "Player not found. Please check the spelling and try again.";
        } else if (httpStatus === 422) {
            if (errorType === "inactive") {
                result.errorMessage =
                    "This player is not currently active in the NBA. Please try another player.";
            } else {
                result.errorMessage =
                    "This player is currently injured and cannot be predicted. Please try another player.";
            }
        } else if (httpStatus >= 500) {
            result.errorMessage =
                "The server encountered an error. Please try again later.";
        }

        return result;
    }

    async function submitPlayerNameAndGetPrediction(rawInput) {
        const cleaned = sanitizePlayerName(rawInput);
        if (!cleaned) {
            return {
                statusCode: 422,
                playerResult: null,
                officialPlayerName: null,
                teamAgainst: null,
                timeAndDateEST: null,
                errorMessage:
                    "Invalid player name.",
            };
        }


        let httpResponse;
        try {
            httpResponse = await sendPredictRequest(cleaned);
        } catch (err) {
            const isTimeout = err.name === "AbortError";
            return {
                statusCode: isTimeout ? 408 : 0,
                playerResult: null,
                officialPlayerName: null,
                teamAgainst: null,
                timeAndDateEST: null,
                errorMessage: isTimeout
                    ? "Request timed out. The server may be busy — please try again."
                    : "Unable to connect to the prediction server. Please check your connection and try again.",
            };
        }

        let body;
        try {
            body = await httpResponse.json();
        } catch (_) {
            return {
                statusCode: httpResponse.status,
                playerResult: null,
                officialPlayerName: null,
                teamAgainst: null,
                timeAndDateEST: null,
                errorMessage: "Received a malformed response from the server.",
            };
        }


        return decodeResponse(httpResponse.status, body);
    }

    window.submitPlayerNameAndGetPrediction = submitPlayerNameAndGetPrediction;
    window.sanitizePlayerName = sanitizePlayerName;
})();
