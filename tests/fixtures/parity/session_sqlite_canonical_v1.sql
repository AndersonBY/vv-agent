PRAGMA user_version = 1;

CREATE TABLE session_items (
    session_id TEXT NOT NULL,
    item_index INTEGER PRIMARY KEY AUTOINCREMENT,
    payload TEXT NOT NULL
);

CREATE INDEX idx_session_items_session_id_item_index
    ON session_items (session_id, item_index);

INSERT INTO session_items (session_id, item_index, payload) VALUES
    ('shared', 3, '{"role":"user","content":"canonical user"}'),
    ('shared', 7, '{"role":"assistant","content":"","tool_calls":[{"id":"call_canonical","type":"function","function":{"name":"lookup","arguments":"{\"a\":1,\"z\":2}"}}]}'),
    ('other', 9, '{"role":"system","content":"canonical other"}');
