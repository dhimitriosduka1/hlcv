from scamp import Session

class GuitarPlayer:
    def __init__(self, tempo=120):
        self.chords = {
            'A': 57,
            'B': 59,
            'C': 60,
            'D': 62,
            'E': 64,
            'F': 65,
            'G': 67
        }
        self.session = Session(tempo=tempo)
        self.instrument = self.session.new_part("classical guitar")
    
    def start(self):
        self.session.start_transcribing()

    def stop(self):
        self.session.stop_transcribing()

    def play(self, chord):
        if chord['note'] not in self.chords:
            raise ValueError(f"Chord '{chord}' is not supported")
        print(f"Playing chord: {chord}")
        self.instrument.play_note(self.chords[chord['note']], volume=chord['volume'], length=chord['length'])

    def play_sequence(self, sequence):
        for chord in sequence:
            self.play(chord)

player = GuitarPlayer()
player.start()

player.play_sequence([
    {'note': 'A', 'volume': 1, 'length': 2},
    {'note': 'C', 'volume': 1, 'length': 2},
    {'note': 'D', 'volume': 1, 'length': 2},
    {'note': 'F', 'volume': 1, 'length': 2},
    {'note': 'E', 'volume': 1, 'length': 2},
    {'note': 'A', 'volume': 1, 'length': 2},
    {'note': 'C', 'volume': 1, 'length': 2},
    {'note': 'D', 'volume': 1, 'length': 2},
    {'note': 'F', 'volume': 1, 'length': 2},
    {'note': 'E', 'volume': 1, 'length': 2}
])
