"""Content-addressed LLM reply cache, scoped to one experiment provenance."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


class RequestCache:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.stats = {"hits": 0, "misses": 0}
        self.used_keys = set()

    @staticmethod
    def _valid_reply(reply):
        # Match generate_cluster_template's parser before persisting a reply.
        # A delimiter-only response must be regenerated on a failed-run retry.
        return isinstance(reply, str) and any(t.strip() for t in reply.split("|||"))

    def get_or_generate(self, request, generate):
        encoded = json.dumps(request, sort_keys=True, ensure_ascii=False, allow_nan=False)
        key = hashlib.sha256(encoded.encode()).hexdigest()
        path = self.directory / (key + ".json")
        self.used_keys.add(key)
        if path.exists():
            record = json.loads(path.read_text())
            reply = record.get("response")
            if (record.get("request") != request or record.get("key") != key
                    or not self._valid_reply(reply)
                    or record.get("response_hash") != hashlib.sha256(reply.encode()).hexdigest()):
                raise ValueError(f"Invalid request cache: {path}")
            self.stats["hits"] += 1
            return reply
        reply = generate()
        if not self._valid_reply(reply):
            raise ValueError("Generation reply without parsed templates cannot be cached")
        self.stats["misses"] += 1
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {"key": key, "request": request, "response": reply,
                   "response_hash": hashlib.sha256(reply.encode()).hexdigest()}
        with tempfile.NamedTemporaryFile(mode="w", dir=self.directory, delete=False) as f:
            json.dump(payload, f, ensure_ascii=False, allow_nan=False)
            temp = Path(f.name)
        os.replace(temp, path)
        return reply

    def attach(self, generator):
        generate = generator.generate_text
        def cached(messages):
            request = {"messages": messages, "model": generator.model_name,
                       "seed": generator.generation_seed,
                       "temperature": generator.generation_temperature,
                       "max_tokens": generator.max_tokens}
            return self.get_or_generate(request, lambda: generate(messages))
        generator.generate_text = cached
