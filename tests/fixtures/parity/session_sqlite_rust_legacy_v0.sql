PRAGMA user_version = 0;

CREATE TABLE session_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    item_json TEXT NOT NULL
);

CREATE INDEX idx_session_items_session_id_id
    ON session_items (session_id, id);

INSERT INTO session_items (id, session_id, item_json) VALUES
    (2, 'shared', '{"type":"user","content":"legacy user"}'),
    (5, 'other', '{"type":"system","content":"other session"}'),
    (8, 'shared', '{"type":"message","message":{"role":"assistant","content":"","tool_calls":[{"id":"call_legacy","type":"function","function":{"name":"lookup","arguments":{"z":1,"a":{"y":2,"x":1}}}}]}}');
