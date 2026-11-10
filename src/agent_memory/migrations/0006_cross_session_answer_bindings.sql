-- A captured cite result carries the placeholder UUID across harness sessions.
-- Keep the event-kind and linear-correction guards; session equality is not identity.
DROP TRIGGER answer_bindings_validate ON answer_bindings;
CREATE TRIGGER answer_bindings_validate AFTER INSERT ON answer_bindings REFERENCING NEW TABLE AS n FOR EACH STATEMENT
    INSERT INTO _append_only_guard
    SELECT error('invalid answer binding: require a placeholder, actual reply, and linear correction')
    FROM n
    JOIN events p ON p.event_id = n.placeholder_event_id
    LEFT JOIN events a ON a.event_id = n.answer_event_id
    LEFT JOIN answer_bindings old ON old.binding_id = n.supersedes_binding_id
    WHERE p.kind <> 'assistant_msg'
       OR NOT coalesce(try_cast(p.payload.placeholder AS BOOLEAN), false)
       OR (n.answer_event_id IS NOT NULL AND
           (a.kind <> 'assistant_msg'
            OR coalesce(try_cast(a.payload.placeholder AS BOOLEAN), false)))
       OR (n.supersedes_binding_id IS NOT NULL
           AND old.placeholder_event_id <> n.placeholder_event_id);
