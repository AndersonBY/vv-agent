PRAGMA user_version = 2;

CREATE TABLE session_items (
    session_id TEXT NOT NULL,
    item_index INTEGER PRIMARY KEY AUTOINCREMENT,
    payload TEXT NOT NULL
);

CREATE INDEX idx_session_items_session_id_item_index
    ON session_items (session_id, item_index);

CREATE TABLE session_commits (
    session_id TEXT NOT NULL,
    commit_id TEXT NOT NULL,
    payload_digest TEXT NOT NULL,
    PRIMARY KEY (session_id, commit_id)
);

INSERT INTO session_items (session_id, item_index, payload) VALUES
    ('shared', 3, '{"role":"user","content":"canonical user"}'),
    ('shared', 7, '{"role":"assistant","content":"","tool_calls":[{"id":"call_canonical","type":"function","function":{"name":"lookup","arguments":"{\"a\":1,\"z\":2}"}}]}'),
    ('shared', 8, '{"role":"tool","content":"<Tool Result Compact>\ntool_name: lookup\nartifact_path: .vv-agent/artifacts/session-1/call-canonical.txt\nretrieval_hint: use read_file on artifact_path if needed\nexcerpt:\nresult: ok\n</Tool Result Compact>","name":"lookup","tool_call_id":"call_canonical","artifact_ref":{"path":".vv-agent/artifacts/session-1/call-canonical.txt","media_type":"text/plain","encoding":"utf-8","size_bytes":26,"sha256":"73c3362aef1718f1c2d34f8691f66e6b5b4b73ba4fce4cdcb2bf27c3f00fea76"}}'),
    ('other', 9, '{"role":"system","content":"canonical other"}');
