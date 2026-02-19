(function (global) {
  'use strict';

  var MAX_LENGTH = 64;
  var ALLOWED_PATTERN = /^[a-zA-Z\s\-']+$/;

  function isControlChar(c) {
    var code = c.charCodeAt(0);
    return (code >= 0 && code <= 31) || code === 127;
  }

  function cleanPlayerName(input) {
    if (input === null || input === undefined) {
      throw new Error('Player name is required');
    }
    if (typeof input !== 'string') {
      throw new Error('Player name must be a string');
    }
    for (var i = 0; i < input.length; i++) {
      if (isControlChar(input[i])) {
        throw new Error('Player name contains invalid control characters');
      }
    }
    var cleaned = input.trim().replace(/\s+/g, ' ');
    if (cleaned.length === 0) {
      throw new Error('Player name is empty after cleaning');
    }
    if (cleaned.length > MAX_LENGTH) {
      throw new Error('Player name must be at most ' + MAX_LENGTH + ' characters');
    }
    if (!ALLOWED_PATTERN.test(cleaned)) {
      throw new Error('Player name may only contain letters, spaces, hyphen, and apostrophe');
    }
    return cleaned;
  }

  function getPredictUrl() {
    var base = (global.PREDICT_API_BASE_URL || '').replace(/\/$/, '') || (global.location ? global.location.origin : 'http://localhost:8000');
    var path = (global.PREDICT_API_ENDPOINT || '/api/predict').replace(/^\/?/, '/');
    return base + path;
  }

  function emptyResult(statusCode, errorMessage) {
    return {
      statusCode: statusCode,
      playerResult: null,
      officialPlayerName: null,
      teamAgainst: null,
      timeAndDateEST: null,
      errorMessage: errorMessage
    };
  }

  function submitPlayerNameAndGetPrediction(rawInput) {
    var cleaned;
    try {
      cleaned = cleanPlayerName(rawInput);
    } catch (err) {
      var msg = err && err.message ? err.message : 'Invalid player name';
      return Promise.resolve(Object.assign(emptyResult(400, msg), { errorMessage: msg }));
    }

    var url = getPredictUrl();
    return fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ playerName: cleaned })
    }).then(function (response) {
      var status = response.status;
      if (status === 404) {
        return response.json().then(function (body) {
          return emptyResult(404, (body && body.errorMessage) ? body.errorMessage : 'Player not found');
        }).catch(function () {
          return emptyResult(404, 'Player not found');
        });
      }
      if (status === 422) {
        return response.json().then(function (body) {
          return emptyResult(422, (body && body.errorMessage) ? body.errorMessage : 'Validation or injury');
        }).catch(function () {
          return emptyResult(422, 'Validation or injury');
        });
      }
      if (status === 200) {
        return response.json().then(function (body) {
          return {
            statusCode: 200,
            playerResult: body.playerResult != null ? body.playerResult : null,
            officialPlayerName: body.officialPlayerName != null ? body.officialPlayerName : null,
            teamAgainst: body.teamAgainst != null ? body.teamAgainst : null,
            timeAndDateEST: body.timeAndDateEST != null ? body.timeAndDateEST : null,
            errorMessage: null
          };
        }).catch(function () {
          return emptyResult(500, 'Invalid response from server');
        });
      }
      return response.json().then(function (body) {
        return emptyResult(status, (body && body.errorMessage) ? body.errorMessage : 'Request failed (' + status + ')');
      }).catch(function () {
        return emptyResult(status, 'Request failed (' + status + ')');
      });
    }).catch(function () {
      return emptyResult(503, 'Network error: could not reach the server');
    });
  }

  global.cleanPlayerName = cleanPlayerName;
  global.submitPlayerNameAndGetPrediction = submitPlayerNameAndGetPrediction;
  global.getPredictUrl = getPredictUrl;
})(typeof window !== 'undefined' ? window : this);
