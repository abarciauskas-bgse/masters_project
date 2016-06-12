import midi, numpy


# analyze song's channels to detect track intervals
# for each track returns the set of notes used
def midi_analyze(midipath):
    pattern = midi.read_midifile(midipath)
    timeleft = [track[0].tick for track in pattern]
    posns = [0 for track in pattern]

    instrument = [0 for track in pattern]
    notes = [set() for track in pattern]
    press = [0 for track in pattern]

    time = 0
    while True:    
        for i in range(len(timeleft)):
            while timeleft[i] == 0:
                track = pattern[i]
                pos = posns[i]
                evt = track[pos]
                if isinstance(evt, midi.NoteEvent):
                    notes[i].add(evt.pitch)
                    press[i] += 1
                try:
                    timeleft[i] = track[pos + 1].tick
                    posns[i] += 1
                except IndexError:
                    timeleft[i] = None

            if timeleft[i] is not None:
                timeleft[i] -= 1

        if all(t is None for t in timeleft):
            break

        time += 1
    np = zip(press, notes, instrument, [i for i in range(len(instrument))])
    np = sorted(np, key=lambda tup: tup[0], reverse = True)
    ranges = []
    tr = []
    for a,b,c,d in np:
        try:
            ranges.append((min(b), max(b)))
            tr.append(d)
        except (ValueError, TypeError):
            pass
    return ranges, tr


# combine tracks from 'setlist' (list of songs) with midi songs to create the training set
# This can take a full folder, or a score-filtered sublist of songs
def process_folder(setlist):
    notesmatrix = []    
    for song in setlist:
        new_notes = create_single_midi_array(song)
    


# compute track decomposition
def split_tracks(ranges):
    lowerNotes, upperNotes = zip(*ranges)
    lowerNotes = list(lowerNotes)
    upperNotes = list(upperNotes)
    span = [upperNotes[i]-lowerNotes[i]+1 for i in range(len(upperNotes))]
    return lowerNotes, upperNotes, span


# returns SIMPLE CHANNEL note-matrix for a coarse time slicing = 16th (always on 4/4)
def create_single_midi_array(midipath):
    lowerNote = 52
    upperNote = 65
    pattern = midi.read_midifile(midipath)

    timeleft = [track[0].tick for track in pattern]
    notesmatrix = []
    time = 0
    span = upperNote - lowerNote

    posns = [0 for track in pattern]
    state = [[0,0] for x in range(span)]
    notesmatrix.append(state)

    while True:
        if time % (pattern.resolution / 4) == (pattern.resolution / 8):
            # Crossed a note boundary. Create a new state, defaulting to holding notes
            oldstate = state
            state = [[oldstate[x][0],0] for x in range(span)]
            notesmatrix.append(state)

        for i in range(len(timeleft)):
            while timeleft[i] == 0:
                track = pattern[i]
                pos = posns[i]
                evt = track[pos]
                if isinstance(evt, midi.NoteEvent):
                    if (evt.pitch < lowerNote) or (evt.pitch >= upperNote):
                        pass # out of bounds
                    else:
                        if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                            state[evt.pitch-lowerNote] = [0, 0]
                        else:
                            state[evt.pitch-lowerNote] = [1, 1]
                elif isinstance(evt, midi.TimeSignatureEvent):
                    if evt.numerator not in (2, 4):
                        return notesmatrix
                try:
                    timeleft[i] = track[pos + 1].tick
                    posns[i] += 1
                except IndexError:
                    timeleft[i] = None

            if timeleft[i] is not None:
                timeleft[i] -= 1

        if all(t is None for t in timeleft):
            break

        time += 1

    return notesmatrix

# returns POLY CHANNEL note-matrix for a coarse time slicing = 16th (always on 4/4)
def create_poly_midi_array(midipath, ranges, tks):
    
    ptn = midi.read_midifile(midipath)

    pattern = midi.Pattern()
    [pattern.append(ptn[i]) for i in tks]
    pattern.resolution = ptn.resolution

    timeleft = [track[0].tick for track in pattern]
    #timeleft = timeleft[:-1]

    notesmatrix = []
    time = 0

    tracks = len(ranges)
    lowerNotes, upperNotes, span = split_tracks(ranges)
    offs = [0 for _ in range(tracks)]

    for i in range(len(span)-1) :
        offs[i+1] = offs[i] + span[i]

    posns = [0 for track in pattern]

    state = [[0,0] for x in range(sum(span))]

    notesmatrix.append(state)

    while len(notesmatrix) < 100*(16*8+1):
        if time % (pattern.resolution / 4) == (pattern.resolution / 8):
            # Crossed a note boundary. Create a new state, defaulting to holding notes
            oldstate = state
            state = [[oldstate[x][0],0] for x in range(sum(span))]
            notesmatrix.append(state)

        for i in range(len(span)):
            while timeleft[i] == 0:
                track = pattern[i]
                pos = posns[i]
                evt = track[pos]
                if isinstance(evt, midi.NoteEvent):
                    if (evt.pitch < lowerNotes[i]) or (evt.pitch > upperNotes[i]):
                        pass # out of bounds
                    else:
                        if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                            state[evt.pitch-lowerNotes[i]+offs[i]] = [0,0]
                        else:
                            state[evt.pitch-lowerNotes[i]+offs[i]] = [1,1]
                elif isinstance(evt, midi.TimeSignatureEvent):
                    if evt.numerator not in (2, 4):
                        return notesmatrix
                try:
                    timeleft[i] = track[pos + 1].tick
                    posns[i] += 1
                except IndexError:
                    timeleft[i] = None

            if timeleft[i] is not None:
                timeleft[i] -= 1

        if all(t is None for t in timeleft):
            break

        time += 1

    return notesmatrix



# write tracks on MIDI file from single-track MIDI
def output_midi_simple_array(notesmatrix, midipath, tickscale = 55):    
    notesmatrix = numpy.asarray(notesmatrix)
    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)
    lowerNote = 52

    e = midi.ProgramChangeEvent()
    e.value = 80
    track.append(e)
    fillTrack(track, notesmatrix, lowerNote, tickscale)    
    
    midi.write_midifile("{}".format(midipath), pattern)



# write tracks on MIDI file from poly-track MIDI
def output_midi_poly_array(notesmatrix, ranges, midipath, tickscale = 55):  
    notesmatrix = numpy.asarray(notesmatrix)
    pattern = midi.Pattern()
    offs = 0
    instrument = [81 for _ in range(len(ranges))]
    for i in range(len(ranges)):
        noffs = offs - ranges[i][0] + ranges[i][1] + 1
        track = midi.Track()
        pattern.append(track)
        lowerNote = ranges[i][0]
        # fill each track at channel 'i'
        fillTrack(track, notesmatrix[:,offs:noffs], lowerNote, tickscale, instrument[i], i)    
        offs = noffs
    
    midi.write_midifile("{}".format(midipath), pattern)




# fill a single track of a multi-track MIDI song
def fillTrack(track, notesmatrix, lowerNote, tickscale, instrument, chn):
    span = len(notesmatrix[0])
    lastcmdtime = 0
    prevstate = [[0,0] for x in range(span)]
    
    e = midi.ProgramChangeEvent(channel=chn)
    e.value = instrument
    track.append(e)

    for time, state in enumerate(notesmatrix + [prevstate[:]]):  
        offNotes = []
        onNotes = []
        for i in range(span):
            n = state[i]
            p = prevstate[i]
            if p[0] == 1:
                if n[0] == 0:
                    offNotes.append(i)
                elif n[1] == 1:
                    offNotes.append(i)
                    onNotes.append(i)
            elif n[0] == 1:
                onNotes.append(i)
        for note in offNotes:
            track.append(midi.NoteOffEvent(tick=(time-lastcmdtime)*tickscale, channel=chn, pitch=note+lowerNote))
            lastcmdtime = time
        for note in onNotes:
            track.append(midi.NoteOnEvent(tick=(time-lastcmdtime)*tickscale, channel=chn, velocity=120, pitch=note+lowerNote))
            lastcmdtime = time
        prevstate = state
    eot = midi.EndOfTrackEvent(tick=1, channel=chn)
    track.append(eot)



def nm_to_input(nm):
    np = numpy.array(nm)
    np_input = numpy.hstack((np[:,:,0],np[:,:,1]))
    return np_input


def output_to_nm(np_input):
    n = len(np_input[0])/2
    a = numpy.dstack((np_input[:,:n],np_input[:,n:]))
    return a.tolist()




if __name__ == "__main__":
    '''
    nm = create_single_midi_array("../musica/2tracks.mid")

    np_input = nm_to_input(nm)
    onm = output_to_nm(np_input)

    ranges = [(52,65),(52,65)]
    nm2 = create_poly_midi_array("../musica/2tracks.mid", ranges)

    output_midi_simple_array(nm, "../musica/2tracks_out.mid")
    output_midi_simple_array(onm, "../musica/2tracks_out_back.mid")

    midipath = "../musica/2tracks_out_poly.mid"
    output_midi_poly_array(nm2, ranges, midipath)
    pattern = midi.read_midifile(midipath)


    ranges = [(40,60)]*5
    nm2 = create_poly_midi_array("../musica/daft_punk-da_funk.mid", ranges)
    nn_input = nm_to_input(nm2)
    onm = output_to_nm(np_input)
    output_midi_simple_array(onm, "../musica/da_funk_out_back.mid")
    '''
