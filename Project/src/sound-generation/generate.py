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

    def play(self, chord, volume, length):
        if chord not in self.chords:
            raise ValueError(f"Chord '{chord}' is not supported")
        print(f"Playing chord: {chord}")
        self.instrument.play_note(self.chords[chord], volume, length)

    def play_sequence(self, sequence, volume, length):
        for pitch in sequence:
            self.play(pitch, volume, length)

player = GuitarPlayer()
player.start()

player.play_sequence(['A', 'B', 'E', 'A', 'G', 'G', 'B', 'C'], 1, 2)
