"""Line-by-line notes for rag/server.py."""

SUMMARY = (
    "The HTTP API and the test page it serves. Seven endpoints, one of them streaming, and "
    "one deliberately disabled by default."
)

BLOCKS = [
    (1, 16,
     "The docstring is the API reference. <b>11</b> flags the one endpoint that is off unless "
     "explicitly enabled. <b>13-15</b> explain the startup design: the pipeline is built once, "
     "because embedding weights take seconds to load and the index is read into memory - "
     "building either per request would dominate the latency of every call."),
    (17, 33,
     "Imports. FastAPI for routing, pydantic for request validation, and the three pieces of the "
     "package the API actually needs."),
    (34, 42,
     "One shared pipeline for the process, created on first use. The global is deliberate: this "
     "object holds the index and the model weights, and a second copy would double the memory "
     "for no benefit."),
    (43, 53,
     "Startup. <b>48-49</b> load the embedder before any request arrives. <b>50-51</b> run a "
     "throwaway search, which is doing more than it looks: it builds the keyword index and loads "
     "the reranker as well, so the first real question is answered at full speed instead of "
     "paying several seconds of model loading while a user waits."),
    (54, 61, "The application, with its lifespan hook attached."),
    (62, 70,
     "The search request shape. Pydantic validates it before any handler runs, so a missing "
     "query or an out-of-range depth is rejected with a clear message rather than failing deeper "
     "in. <b>68-69</b> note the three-state rerank flag: null means \"use the configured "
     "default\", which is different from false."),
    (71, 81,
     "The ask request. <b>74-76</b> record why the default depth is null rather than a number: "
     "letting the token budget decide how many passages fit measured better than any fixed "
     "count, so the API's default is the measured-best behaviour and a number is an override."),
    (82, 89, "The ingest request, and the path to the test page."),
    (90, 95, "Serving the page itself at the root, which is what makes the serve command useful."),
    (96, 106,
     "Example questions for the page. They are read from the evaluation set, so the page opens "
     "with questions that are <i>known</i> to have answers in the indexed documents - a first "
     "impression that demonstrates the system rather than testing the user's luck. <b>105-106</b> "
     "return nothing when there is no evaluation set, and the page simply shows no examples."),
    (107, 125,
     "Choosing which examples to show. <b>117-121</b> take one question of each kind first, so "
     "the page displays the range of what the system handles - Arabic, English, a direct article "
     "lookup - rather than four variations of one thing. <b>122-123</b> fill the remaining slots "
     "without repeating."),
    (126, 136,
     "Health and stats. Health answers the two questions a monitor cares about: is it up, and "
     "does it have anything indexed."),
    (137, 148, "Search, which is the pipeline call plus serialisation."),
    (149, 164,
     "Ask. <b>161-162</b> translate a provider failure into a 502 - the upstream model failed, "
     "not this service - carrying the explanatory message from the backend so the caller sees "
     "\"Gemini rejected the API key\" rather than a bare status code."),
    (165, 177,
     "The streaming endpoint. The sources are resolved <i>before</i> the generator is created, "
     "which is what allows them to be sent first."),
    (178, 188,
     "The event stream. <b>179-180</b> send the sources as the opening event, so the page can "
     "render the citation list while the answer is still being written. <b>182-183</b> send each "
     "text fragment as it arrives. <b>184-186</b> turn a mid-stream failure into an error event "
     "rather than a truncated response that would look like a complete answer. <b>187</b> closes "
     "the stream explicitly, so the client knows the difference between finished and "
     "disconnected."),
    (189, 195,
     "The streaming response headers. <b>192</b> disables proxy buffering, without which a "
     "reverse proxy would hold the whole answer and deliver it in one block - which defeats the "
     "entire point of streaming."),
    (196, 209,
     "Ingestion over HTTP, off by default. The docstring is explicit about why: this endpoint "
     "reads arbitrary paths on the server's filesystem, which is a real capability to hand to an "
     "HTTP caller. <b>204-208</b> refuse unless the environment variable is set, and the message "
     "names the variable - so enabling it is a deliberate act, behind whatever authentication "
     "the deployment already has."),
    (210, 216, "The ingest result, summarised the same way the CLI summarises it."),
]
