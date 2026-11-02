"""Line-by-line notes for rag/llm.py."""

SUMMARY = (
    "The only part of the system that talks to a hosted model, and the only part that knows "
    "which provider is in use. Two backends - Claude and Gemini - behind one interface, so the "
    "provider can be swapped without re-indexing anything."
)

BLOCKS = [
    (1, 16,
     "The docstring states the boundary this module exists to draw. <b>3-5</b>: only generation "
     "is hosted - loading, chunking, embedding, keyword search and reranking all run locally, so "
     "switching provider changes nothing about the index. <b>7-12</b> define the neutral request "
     "shape and name what is kept inside each backend: cache breakpoints, role names, thinking "
     "levels. Because those stay here, the prompt builder, the citation parser and the CLI never "
     "branch on which provider is configured."),
    (17, 25, "Imports. Neither provider SDK is imported here - each is loaded inside its own backend."),
    (26, 32,
     "Default models and the environment variables each provider's credentials can arrive in. "
     "<b>27</b> records something learned by trying: the 2.5 Gemini models are restricted to "
     "accounts that already used them, so a new key gets a 404 from them and the 3.x default "
     "avoids that."),
    (33, 42,
     "One message shown whenever credentials are missing, wherever that is discovered. It names "
     "both options, links the free one, and ends with the fact that saves a confused user: "
     "retrieval needs no key at all."),
    (43, 52,
     "The provider-neutral result. <b>50-51</b> carry refusal as data rather than as an "
     "exception, because a refusal is a legitimate outcome that the caller should display, not "
     "an error to be caught."),
    (53, 75,
     "The interface both backends implement: complete, stream, and a method that turns one of "
     "the SDK's exceptions into a sentence a user can act on. Written as a Protocol, so neither "
     "backend inherits anything."),
    (76, 83, "The Claude backend begins."),
    (84, 99,
     "Construction. <b>92-94</b> record a subtlety worth knowing: the Anthropic SDK resolves a "
     "key from the API key variable, then an auth token, then a logged-in profile - so an unset "
     "environment variable is <i>not</i> proof that there are no credentials. The code lets the "
     "SDK decide and translates its failure, rather than guessing from the environment."),
    (100, 114,
     "Building the request. <b>102-105</b> place a cache breakpoint after the retrieved context, "
     "so a follow-up question over the same passages is billed at the cached rate. <b>109</b> "
     "uses adaptive thinking, letting the model decide how much reasoning a question needs. "
     "<b>110</b> maps the configured effort onto the output configuration."),
    (115, 126,
     "The non-streaming call, which still streams internally. <b>118-121</b> carry the reason in "
     "a comment: a long answer requested in one shot can exceed the request timeout, while a "
     "streamed one keeps the connection alive and is assembled at the end. <b>124-125</b> convert "
     "any SDK exception into a readable message."),
    (127, 139,
     "Reading the response. <b>130-132</b> concatenate only the text blocks, since the response "
     "can also contain thinking blocks that are not part of the answer. <b>127</b> and "
     "<b>137</b> capture a refusal and its category, which the CLI shows instead of pretending "
     "an empty answer was produced."),
    (140, 147,
     "Streaming, which is the same request yielding text as it arrives."),
    (148, 168,
     "Error translation, ordered from most specific to least. Each branch answers the only "
     "question a user has at that moment: <b>150-153</b> no usable credentials, <b>154-155</b> "
     "valid credentials without access to this model, <b>156-157</b> a model id that does not "
     "exist and the setting that holds it, <b>158-160</b> a rate limit with the retry delay read "
     "from the response headers, <b>161-162</b> no network. <b>167</b> falls back to the raw "
     "error rather than hiding an unanticipated failure behind a generic message."),
    (169, 180,
     "Usage figures, read defensively with <font name='Mono' size='8'>getattr</font> so an SDK "
     "that stops reporting a field degrades the statistics instead of raising. <b>179</b> drops "
     "the missing ones, so the output shows what is actually known."),
    (181, 197,
     "The Gemini section opens with the two mappings that differ from Claude. <b>186-194</b> map "
     "the effort scale onto thinking levels, collapsing the top three to \"high\" because that is "
     "the ceiling here - and the comment records that Gemini 3.x always thinks and can only be "
     "turned down, not off. <b>196</b> lists the finish reasons that mean the model declined "
     "rather than answered."),
    (198, 211,
     "Construction, with its own lazy import so a user of one provider never needs the other's "
     "SDK installed."),
    (212, 220,
     "Credential checks. <b>212-214</b> accept either an API key or Vertex AI configuration, "
     "failing early with the shared message if neither is present - which turns a confusing SDK "
     "error into one that names the variable to set."),
    (221, 241,
     "Translating the neutral request into this provider's shape. <b>225-226</b> rename the "
     "assistant role to \"model\", which is this API's name for the same thing. <b>233</b> joins "
     "the system parts, since this API takes one instruction rather than a list. <b>236-238</b> "
     "disable automatic function calling: no tools are declared, and leaving it enabled makes "
     "the SDK warn on every single call."),
    (242, 248,
     "Which failures are worth retrying: server errors, which clear on their own, and rate "
     "limits. Ordinary client errors are not retried, because a malformed request or a bad key "
     "will fail identically every time."),
    (249, 266,
     "The retry loop. The docstring records why it was added rather than theorised: this "
     "provider returns 503 when busy, and it happened during real use here. <b>264-265</b> "
     "double the delay after each attempt, so two retries wait 1.5 and 3 seconds rather than "
     "hammering a service that is already overloaded. <b>262-263</b> stop immediately for "
     "anything not retryable."),
    (267, 283,
     "The call and its response. <b>271-274</b> check for refusal in both places this API can "
     "report it - a blocked prompt and a refusing finish reason. <b>273</b> takes the text after "
     "the last dot, because the finish reason may arrive as an enum whose string form is "
     "qualified. <b>279</b> maps the finish reason onto the neutral vocabulary, so a caller can "
     "tell a truncated answer from a complete one without knowing whose API produced it."),
    (284, 304,
     "Streaming, with a distinction that matters. <b>286-288</b>: only the <i>opening</i> of the "
     "stream is retried, because once text has reached the caller a restart would repeat it. "
     "<b>299-303</b> handle a failure mid-stream - which happened in real use - by prefixing the "
     "message with \"the answer was cut short\", so a user who already has half an answer on "
     "screen understands what they are looking at rather than assuming it is complete."),
    (305, 319,
     "Error translation for this provider. <b>312-319</b> exist because of a real diagnostic "
     "trap: an invalid key is reported as a 400, not a 401, so without this branch the message "
     "would be a generic bad-request error. Instead it says a key was found and is wrong, names "
     "the file to check, notes that the environment overrides that file, and links where to get "
     "a new one."),
    (320, 335,
     "The remaining cases: rejected credentials, a model this key cannot use - with a current "
     "model named, since that was the actual cause when it occurred - a quota limit, and server "
     "errors."),
    (336, 348,
     "Usage figures again, under this provider's field names. <b>344</b> reports thinking tokens, "
     "which have no Claude equivalent and are worth seeing because they are billed."),
    (349, 356,
     "The registry and a backend cache keyed by provider and model, so repeated queries in one "
     "process do not reconstruct a client each time."),
    (357, 368,
     "Detecting credentials without constructing anything. <b>359-361</b> also look for a "
     "logged-in Anthropic profile directory, because of the resolution order noted at "
     "<b>92-94</b> - checking only environment variables would report \"no credentials\" for a "
     "machine that is in fact logged in."),
    (369, 385,
     "Provider resolution. <b>372-377</b> accept an explicit choice and reject an unknown one by "
     "listing the valid values. <b>378-383</b> are the automatic path, and <b>379-380</b> "
     "explain the order: Gemini is tried first because it is the one with a free tier, so a key "
     "for it is the more likely deliberate choice when both exist."),
    (386, 396,
     "Getting a backend, and the default model for a provider - used by the CLI to print what "
     "will actually be called before anything is called."),
]
