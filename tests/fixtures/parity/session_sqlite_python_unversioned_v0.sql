PRAGMA user_version = 0;

CREATE TABLE session_items (
    session_id TEXT NOT NULL,
    item_index INTEGER PRIMARY KEY AUTOINCREMENT,
    payload TEXT NOT NULL
);

INSERT INTO session_items (session_id, item_index, payload) VALUES
    ('shared', 4, '{"role":"user","content":"python unversioned"}');
