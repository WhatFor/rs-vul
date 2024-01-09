use std::fmt::{Debug, Error, Formatter};

// todo: restructure MidiEvent into an enum with data on it
//       so that we can have a MidiEvent::NoteOn(note, velocity) and MidiEvent::NoteOff(note)

#[derive(PartialEq)]
pub enum MidiEventType {
    NoteOn,
    NoteOff,
}

pub struct MidiEvent {
    pub event_type: MidiEventType,
    pub channel: u8,
    pub note: u8,
    pub velocity: Option<u8>,
}

impl Debug for MidiEventType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            MidiEventType::NoteOn => write!(f, "ON"),
            MidiEventType::NoteOff => write!(f, "OFF"),
        }
    }
}

impl MidiEvent {
    pub fn new(
        event_type: MidiEventType,
        channel: u8,
        note: u8,
        velocity: Option<u8>,
    ) -> MidiEvent {
        MidiEvent {
            event_type,
            channel,
            note,
            velocity,
        }
    }
}

impl Debug for MidiEvent {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(
            f,
            "MidiEvent {{ event_type: {:?}, channel: {}, note: {}, velocity: {} }}",
            self.event_type,
            self.channel,
            self.note,
            self.velocity.unwrap_or(0)
        )
    }
}
