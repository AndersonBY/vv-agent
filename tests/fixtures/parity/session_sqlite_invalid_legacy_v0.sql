PRAGMA user_version = 0;

CREATE TABLE session_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    item_json TEXT NOT NULL
);

INSERT INTO session_items (id, session_id, item_json) VALUES
    (1, 'shared', '{"type":"user","content":"valid first row"}'),
    (2, 'shared', '{"type":"unknown","content":"invalid second row"}');
