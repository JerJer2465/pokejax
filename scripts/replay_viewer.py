#!/usr/bin/env python3
"""Pokejax Battle Replay Viewer — GUI for debugging the engine.

Loads .pkl replay files recorded by scripts/record_replay.py and displays
them as an interactive battle viewer. Runs on Windows Python (no JAX needed).

Usage:
    python scripts/replay_viewer.py                          # file dialog
    python scripts/replay_viewer.py replays/game_0000.pkl   # direct load
"""

import argparse
import os
import sys
import pickle
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

# ---------------------------------------------------------------------------
# Display constants
# ---------------------------------------------------------------------------
STATUS_LABELS = {0: '', 1: 'BRN', 2: 'PSN', 3: 'TOX', 4: 'SLP', 5: 'FRZ', 6: 'PAR'}
STATUS_COLORS = {0: '', 1: '#e06030', 2: '#9060c0', 3: '#9060c0',
                 4: '#6090e0', 5: '#90c0ff', 6: '#f0d040'}

WEATHER_LABELS = {0: 'None', 1: 'Harsh Sunlight', 2: 'Rain',
                  3: 'Sandstorm', 4: 'Hail'}
TERRAIN_LABELS = {0: 'None', 1: 'Electric Terrain', 2: 'Grassy Terrain',
                  3: 'Misty Terrain', 4: 'Psychic Terrain'}

SC_NAMES = [
    'Spikes', 'Toxic Spikes', 'Stealth Rock', 'Sticky Web',
    'Reflect', 'Light Screen', 'Aurora Veil', 'Tailwind',
    'Safeguard', 'Mist',
]

BOOST_NAMES = ['Atk', 'Def', 'SpA', 'SpD', 'Spe', 'Acc', 'Eva']

TYPE_NAMES = [
    '???', 'Normal', 'Fire', 'Water', 'Electric', 'Grass', 'Ice',
    'Fighting', 'Poison', 'Ground', 'Flying', 'Psychic', 'Bug',
    'Rock', 'Ghost', 'Dragon', 'Dark', 'Steel', 'Fairy',
]

TYPE_COLORS = {
    'Normal': '#a8a878', 'Fire': '#f08030', 'Water': '#6890f0',
    'Electric': '#f8d030', 'Grass': '#78c850', 'Ice': '#98d8d8',
    'Fighting': '#c03028', 'Poison': '#a040a0', 'Ground': '#e0c068',
    'Flying': '#a890f0', 'Psychic': '#f85888', 'Bug': '#a8b820',
    'Rock': '#b8a038', 'Ghost': '#705898', 'Dragon': '#7038f8',
    'Dark': '#705848', 'Steel': '#b8b8d0', 'Fairy': '#ee99ac',
    '???': '#888888',
}

VOL_DISPLAY = {
    0:  ('Confused',    True),
    1:  ('Flinch',      False),
    2:  ('Trapped',     True),
    3:  ('Seeded',      False),
    4:  ('Substitute',  True),
    5:  ('Protect',     False),
    6:  ('Encore',      True),
    7:  ('Taunt',       True),
    8:  ('Torment',     False),
    9:  ('Disable',     True),
    10: ('Endure',      False),
    11: ('Magic Coat',  False),
    12: ('Snatch',      False),
    13: ('Ingrain',     False),
    15: ('Heal Block',  True),
    16: ('Embargo',     True),
    17: ('Charging',    False),
    18: ('Recharging',  False),
    19: ('Locked',      True),
    20: ('Choice Lock', True),
    21: ('Focus Nrg',   False),
    22: ('Minimize',    False),
    23: ('Curse',       False),
    24: ('Nightmare',   False),
    25: ('Infatuated',  False),
    26: ('Yawn',        True),
    27: ('Destiny Bnd', False),
    28: ('Grudge',      False),
    30: ('Perish Song', True),
}

# Event tag → color mapping
EVENT_COLORS = {
    'action': '#ddeeff',
    'damage': '#ff9999',
    'heal':   '#88dd88',
    'faint':  '#ff4444',
    'status': '#ffbb66',
    'boost':  '#99ccff',
    'field':  '#cc99ff',
}

# Dark theme palette
BG         = '#1a1a2e'
BG_CARD    = '#16213e'
BG_PANEL   = '#0f3460'
FG         = '#e0e0e0'
FG_DIM     = '#888899'
ACCENT     = '#533483'
BORDER     = '#533483'
FONT_MONO  = ('Consolas', 9)
FONT_LABEL = ('Segoe UI', 9)
FONT_TITLE = ('Segoe UI', 10, 'bold')
FONT_SMALL = ('Consolas', 8)


# ---------------------------------------------------------------------------
# Helper functions (no JAX, no pokejax imports)
# ---------------------------------------------------------------------------

def _name(snap, side, slot, tables, key='species_id', names_key='species_names'):
    idx = int(snap[key][side, slot])
    names = tables[names_key]
    if 0 < idx < len(names):
        return names[idx]
    return f'???({idx})'


def _species_name(snap, side, slot, tables):
    return _name(snap, side, slot, tables)


def _move_name(snap, side, slot, move_slot, tables):
    mid = int(snap['move_ids'][side, slot, move_slot])
    names = tables['move_names']
    if 0 < mid < len(names):
        return names[mid]
    return '???'


def _ability_name(snap, side, slot, tables):
    return _name(snap, side, slot, tables, 'ability_id', 'ability_names')


def _item_name(snap, side, slot, tables):
    idx = int(snap['item_id'][side, slot])
    names = tables['item_names']
    if 0 < idx < len(names):
        return names[idx]
    return ''


def _hp_color(hp, max_hp):
    frac = hp / max(max_hp, 1)
    if frac > 0.5:
        return '#44cc44'
    elif frac > 0.25:
        return '#ddcc00'
    else:
        return '#cc3333'


def decode_volatiles(vol_mask: int, vol_data: np.ndarray) -> str:
    parts = []
    for bit, (name, has_ctr) in VOL_DISPLAY.items():
        if vol_mask & (1 << bit):
            if has_ctr and vol_data is not None:
                ctr = int(vol_data[bit])
                parts.append(f'{name}({ctr})')
            else:
                parts.append(name)
    return ', '.join(parts) if parts else '—'


def format_boosts(boosts_arr) -> str:
    parts = []
    for i, name in enumerate(BOOST_NAMES):
        b = int(boosts_arr[i])
        if b > 0:
            parts.append(f'+{b} {name}')
        elif b < 0:
            parts.append(f'{b} {name}')
    return ', '.join(parts) if parts else '—'


def format_side_conditions(sc_arr) -> str:
    parts = []
    for i, cname in enumerate(SC_NAMES):
        v = int(sc_arr[i])
        if v == 0:
            continue
        if i <= 3:  # layer-count conditions
            parts.append(f'{cname}×{v}')
        else:
            parts.append(f'{cname}({v}t)')
    return ', '.join(parts) if parts else 'None'


# ---------------------------------------------------------------------------
# HP Bar widget
# ---------------------------------------------------------------------------

class HPBar:
    def __init__(self, parent, width=180, height=12):
        self.width = width
        self.height = height
        self.canvas = tk.Canvas(parent, width=width, height=height,
                                bg='#333344', highlightthickness=1,
                                highlightbackground=BORDER)
        self._bar = self.canvas.create_rectangle(0, 0, 0, height, fill='#44cc44')

    def update(self, hp: int, max_hp: int):
        frac = max(0.0, hp / max(max_hp, 1))
        color = _hp_color(hp, max_hp)
        filled = int(frac * self.width)
        self.canvas.coords(self._bar, 0, 0, filled, self.height)
        self.canvas.itemconfig(self._bar, fill=color)
        self.canvas.configure(bg='#222233' if hp <= 0 else '#333344')


# ---------------------------------------------------------------------------
# Pokemon slot row (in team roster)
# ---------------------------------------------------------------------------

class RosterSlot:
    def __init__(self, parent, row):
        self.frame = tk.Frame(parent, bg=BG_CARD, pady=1)
        self.frame.grid(row=row, column=0, sticky='ew', padx=2, pady=1)
        self.frame.columnconfigure(1, weight=1)

        self.lbl_name = tk.Label(self.frame, text='', width=14, anchor='w',
                                 bg=BG_CARD, fg=FG, font=FONT_MONO)
        self.lbl_name.grid(row=0, column=0, sticky='w', padx=(2, 4))

        self.hp_bar = HPBar(self.frame, width=80, height=8)
        self.hp_bar.canvas.grid(row=0, column=1, sticky='w')

        self.lbl_hp = tk.Label(self.frame, text='', width=8, anchor='w',
                               bg=BG_CARD, fg=FG_DIM, font=FONT_SMALL)
        self.lbl_hp.grid(row=0, column=2, sticky='w', padx=(4, 0))

        self.lbl_status = tk.Label(self.frame, text='', width=4, anchor='center',
                                   bg=BG_CARD, font=FONT_SMALL)
        self.lbl_status.grid(row=0, column=3, sticky='w', padx=(2, 2))

    def update(self, snap, side, slot, tables, is_active):
        sid = int(snap['species_id'][side, slot])
        if sid <= 0:
            self.lbl_name.configure(text='—', fg=FG_DIM)
            self.hp_bar.update(0, 1)
            self.lbl_hp.configure(text='')
            self.lbl_status.configure(text='', bg=BG_CARD)
            return

        name = _species_name(snap, side, slot, tables)
        hp   = int(snap['hp'][side, slot])
        mhp  = int(snap['max_hp'][side, slot])
        fnt  = bool(snap['fainted'][side, slot])
        st   = int(snap['status'][side, slot])

        # Name styling
        prefix = '★' if is_active else ' '
        fg_name = FG_DIM if fnt else (ACCENT if is_active else FG)
        self.lbl_name.configure(text=f'{prefix}{name[:12]}', fg=fg_name,
                                 font=('Consolas', 9, 'bold') if is_active else FONT_MONO)
        self.hp_bar.update(0 if fnt else hp, mhp)
        hp_text = 'FNT' if fnt else f'{hp}/{mhp}'
        self.lbl_hp.configure(text=hp_text, fg='#666677' if fnt else FG_DIM)
        st_label = STATUS_LABELS[st]
        st_color = STATUS_COLORS[st]
        self.lbl_status.configure(text=st_label,
                                   fg=st_color if st_color else FG_DIM,
                                   bg=BG_CARD)


# ---------------------------------------------------------------------------
# Active Pokemon panel
# ---------------------------------------------------------------------------

class ActivePokemonPanel:
    def __init__(self, parent):
        self.frame = tk.LabelFrame(parent, text='Active Pokémon', bg=BG_PANEL,
                                   fg=ACCENT, font=FONT_TITLE,
                                   relief='flat', bd=1)

        # Row 0: Name + types
        r = tk.Frame(self.frame, bg=BG_PANEL)
        r.grid(row=0, column=0, sticky='ew', padx=4, pady=(4, 2))
        self.lbl_name = tk.Label(r, text='', anchor='w', bg=BG_PANEL, fg=FG,
                                 font=('Consolas', 11, 'bold'))
        self.lbl_name.pack(side='left')
        self.lbl_types = tk.Label(r, text='', anchor='w', bg=BG_PANEL, fg=FG_DIM,
                                  font=FONT_LABEL)
        self.lbl_types.pack(side='left', padx=(6, 0))

        # Row 1: HP bar + text
        r = tk.Frame(self.frame, bg=BG_PANEL)
        r.grid(row=1, column=0, sticky='ew', padx=4, pady=2)
        self.hp_bar = HPBar(r, width=200, height=16)
        self.hp_bar.canvas.pack(side='left')
        self.lbl_hp = tk.Label(r, text='', bg=BG_PANEL, fg=FG, font=FONT_MONO)
        self.lbl_hp.pack(side='left', padx=(8, 0))

        # Row 2: Status + Level + ability + item in one line
        self.lbl_info = tk.Label(self.frame, text='', anchor='w', bg=BG_PANEL,
                                 fg=FG_DIM, font=FONT_SMALL)
        self.lbl_info.grid(row=2, column=0, sticky='ew', padx=4, pady=1)

        # Row 3: Boosts
        self.lbl_boosts = tk.Label(self.frame, text='', anchor='w', bg=BG_PANEL,
                                   fg='#99ccff', font=FONT_SMALL)
        self.lbl_boosts.grid(row=3, column=0, sticky='ew', padx=4, pady=1)

        # Row 4: Volatiles
        self.lbl_vols = tk.Label(self.frame, text='', anchor='w', bg=BG_PANEL,
                                 fg='#ffcc88', font=FONT_SMALL, wraplength=280)
        self.lbl_vols.grid(row=4, column=0, sticky='ew', padx=4, pady=1)

        # Row 5-8: Move rows
        moves_frame = tk.Frame(self.frame, bg=BG_PANEL)
        moves_frame.grid(row=5, column=0, sticky='ew', padx=4, pady=(4, 2))
        tk.Label(moves_frame, text='Moves:', bg=BG_PANEL, fg=ACCENT,
                 font=FONT_TITLE).grid(row=0, column=0, sticky='w', pady=(0, 2))

        self.move_rows = []
        for mi in range(4):
            mf = tk.Frame(moves_frame, bg=BG_CARD)
            mf.grid(row=mi + 1, column=0, sticky='ew', pady=1)
            lbl = tk.Label(mf, text='', width=22, anchor='w',
                           bg=BG_CARD, fg=FG, font=FONT_MONO)
            lbl.grid(row=0, column=0, sticky='w', padx=2)
            pp_bar = HPBar(mf, width=80, height=8)
            pp_bar.canvas.grid(row=0, column=1, sticky='w', padx=(4, 2))
            pp_lbl = tk.Label(mf, text='', width=7, anchor='w',
                              bg=BG_CARD, fg=FG_DIM, font=FONT_SMALL)
            pp_lbl.grid(row=0, column=2, sticky='w')
            self.move_rows.append((lbl, pp_bar, pp_lbl))

        self.frame.columnconfigure(0, weight=1)

    def update(self, snap, side, tables):
        active = int(snap['active_idx'][side])
        sid = int(snap['species_id'][side, active])

        if sid <= 0:
            self.lbl_name.configure(text='—')
            self.lbl_types.configure(text='')
            self.hp_bar.update(0, 1)
            self.lbl_hp.configure(text='—/—')
            self.lbl_info.configure(text='')
            self.lbl_boosts.configure(text='')
            self.lbl_vols.configure(text='')
            for lbl, pb, pl in self.move_rows:
                lbl.configure(text='—')
                pb.update(0, 1)
                pl.configure(text='')
            return

        name = _species_name(snap, side, active, tables)
        hp   = int(snap['hp'][side, active])
        mhp  = int(snap['max_hp'][side, active])
        st   = int(snap['status'][side, active])
        lv   = int(snap['level'][side, active])

        types_raw = [int(snap['types'][side, active, t]) for t in range(2)]
        type_strs  = [TYPE_NAMES[t] for t in types_raw if t > 0]
        types_str  = ' / '.join(type_strs)

        ability = _ability_name(snap, side, active, tables)
        item    = _item_name(snap, side, active, tables)

        st_label = STATUS_LABELS[st]
        boosts_arr = snap['boosts'][side, active]
        boosts_str = format_boosts(boosts_arr)

        vol_mask = int(snap['volatiles'][side, active])
        vol_data = snap['volatile_data'][side, active] if 'volatile_data' in snap else None
        vols_str = decode_volatiles(vol_mask, vol_data)

        fnt = bool(snap['fainted'][side, active])
        fg_name = '#cc3333' if fnt else FG

        self.lbl_name.configure(text=name, fg=fg_name)
        self.lbl_types.configure(text=f'[{types_str}]')
        self.hp_bar.update(0 if fnt else hp, mhp)
        hp_pct = hp / max(mhp, 1) * 100
        self.lbl_hp.configure(text=f'{hp}/{mhp} ({hp_pct:.0f}%)',
                               fg=_hp_color(hp, mhp))

        info_parts = [f'Lv{lv}']
        if st_label:
            info_parts.append(st_label)
        if ability:
            info_parts.append(ability)
        if item:
            info_parts.append(f'@{item}')
        self.lbl_info.configure(text='  '.join(info_parts))
        self.lbl_boosts.configure(text=f'Boosts: {boosts_str}')
        self.lbl_vols.configure(text=f'Vols: {vols_str}')

        # Move rows
        for mi, (lbl, pp_bar, pp_lbl) in enumerate(self.move_rows):
            mid = int(snap['move_ids'][side, active, mi])
            if mid <= 0:
                lbl.configure(text='—', fg=FG_DIM)
                pp_bar.update(0, 1)
                pp_lbl.configure(text='')
                continue
            mname  = _move_name(snap, side, active, mi, tables)
            pp     = int(snap['move_pp'][side, active, mi])
            max_pp = int(snap['move_max_pp'][side, active, mi])
            disabled = bool(snap['move_disabled'][side, active, mi]) if 'move_disabled' in snap else False
            fg_move = '#666677' if disabled else FG
            suffix = ' [DIS]' if disabled else ''
            lbl.configure(text=f'{mname[:18]}{suffix}', fg=fg_move)
            pp_bar.update(pp, max(max_pp, 1))
            pp_lbl.configure(text=f'PP {pp}/{max_pp}')


# ---------------------------------------------------------------------------
# Team panel (active + roster)
# ---------------------------------------------------------------------------

class TeamPanel:
    def __init__(self, parent, side_label):
        self.frame = tk.Frame(parent, bg=BG, relief='flat')
        self.frame.columnconfigure(0, weight=1)

        # Player name header
        self.lbl_player = tk.Label(self.frame, text=side_label, bg=BG,
                                   fg=ACCENT, font=('Segoe UI', 12, 'bold'))
        self.lbl_player.grid(row=0, column=0, sticky='ew', padx=4, pady=(4, 2))

        # Pokemon count
        self.lbl_count = tk.Label(self.frame, text='', bg=BG, fg=FG_DIM,
                                  font=FONT_SMALL)
        self.lbl_count.grid(row=1, column=0, sticky='ew', padx=4, pady=(0, 4))

        # Active Pokemon panel
        self.active_panel = ActivePokemonPanel(self.frame)
        self.active_panel.frame.grid(row=2, column=0, sticky='ew', padx=4, pady=4)

        # Roster
        roster_frame = tk.LabelFrame(self.frame, text='Team', bg=BG_CARD,
                                     fg=ACCENT, font=FONT_TITLE,
                                     relief='flat', bd=1)
        roster_frame.grid(row=3, column=0, sticky='ew', padx=4, pady=4)
        roster_frame.columnconfigure(0, weight=1)

        self.roster_slots = []
        for i in range(6):
            slot = RosterSlot(roster_frame, row=i)
            slot.frame.columnconfigure(1, weight=1)
            self.roster_slots.append(slot)

    def update(self, snap, side, tables, player_name):
        self.lbl_player.configure(text=player_name)
        left = int(snap['pokemon_left'][side])
        self.lbl_count.configure(text=f'{left}/6 remaining')
        self.active_panel.update(snap, side, tables)
        active_idx = int(snap['active_idx'][side])
        for i, slot in enumerate(self.roster_slots):
            slot.update(snap, side, i, tables, is_active=(i == active_idx))


# ---------------------------------------------------------------------------
# Field + Events center panel
# ---------------------------------------------------------------------------

class CenterPanel:
    def __init__(self, parent):
        self.frame = tk.Frame(parent, bg=BG)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(1, weight=1)

        # Field state
        field_frame = tk.LabelFrame(self.frame, text='Field', bg=BG_PANEL,
                                    fg=ACCENT, font=FONT_TITLE,
                                    relief='flat', bd=1)
        field_frame.grid(row=0, column=0, sticky='ew', padx=4, pady=4)
        field_frame.columnconfigure(1, weight=1)

        labels = ['Weather:', 'Terrain:', 'Trick Room:', 'P1 Side:', 'P2 Side:']
        self.field_vals = []
        for i, lbl in enumerate(labels):
            tk.Label(field_frame, text=lbl, bg=BG_PANEL, fg=FG_DIM,
                     font=FONT_MONO, anchor='e', width=11).grid(
                         row=i, column=0, sticky='e', padx=(4, 2), pady=1)
            v = tk.Label(field_frame, text='—', bg=BG_PANEL, fg=FG,
                         font=FONT_MONO, anchor='w')
            v.grid(row=i, column=1, sticky='w', padx=(2, 4), pady=1)
            self.field_vals.append(v)

        # Events log
        events_frame = tk.LabelFrame(self.frame, text='Turn Events', bg=BG_CARD,
                                     fg=ACCENT, font=FONT_TITLE,
                                     relief='flat', bd=1)
        events_frame.grid(row=1, column=0, sticky='nsew', padx=4, pady=4)
        events_frame.columnconfigure(0, weight=1)
        events_frame.rowconfigure(0, weight=1)

        self.events_text = tk.Text(events_frame, bg='#111122', fg=FG,
                                   font=FONT_MONO, state='disabled',
                                   relief='flat', wrap='word',
                                   height=20)
        self.events_text.grid(row=0, column=0, sticky='nsew', padx=2, pady=2)

        scrollbar = ttk.Scrollbar(events_frame, orient='vertical',
                                  command=self.events_text.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        self.events_text.configure(yscrollcommand=scrollbar.set)

        # Configure event tags
        self.events_text.tag_configure('action', foreground='#ddeeff')
        self.events_text.tag_configure('damage', foreground='#ff9999')
        self.events_text.tag_configure('heal',   foreground='#88ee88')
        self.events_text.tag_configure('faint',  foreground='#ff4444',
                                       font=('Consolas', 9, 'bold'))
        self.events_text.tag_configure('status', foreground='#ffbb66')
        self.events_text.tag_configure('boost',  foreground='#99ccff')
        self.events_text.tag_configure('field',  foreground='#cc99ff')

    def update_field(self, snap, player_names):
        weather = int(snap['weather'])
        w_turns = int(snap['weather_turns'])
        w_label = WEATHER_LABELS.get(weather, '—')
        if weather > 0 and w_turns > 0:
            w_label += f' ({w_turns}t)'

        terrain = int(snap.get('terrain', 0))
        t_turns = int(snap.get('terrain_turns', 0))
        t_label = TERRAIN_LABELS.get(terrain, '—')
        if terrain > 0 and t_turns > 0:
            t_label += f' ({t_turns}t)'

        tr = int(snap['trick_room'])
        tr_label = f'ON ({tr}t)' if tr > 0 else 'Off'
        tr_color = '#ff88ff' if tr > 0 else FG_DIM

        sc0 = format_side_conditions(snap['side_conditions'][0])
        sc1 = format_side_conditions(snap['side_conditions'][1])

        vals = [w_label, t_label, tr_label,
                f'{player_names[0]}: {sc0}', f'{player_names[1]}: {sc1}']
        colors = [FG, FG, tr_color, FG, FG]
        for widget, val, col in zip(self.field_vals, vals, colors):
            widget.configure(text=val, fg=col)

    def update_events(self, events: list):
        self.events_text.configure(state='normal')
        self.events_text.delete('1.0', 'end')
        for tag, text in events:
            self.events_text.insert('end', f'• {text}\n', tag)
        self.events_text.configure(state='disabled')
        self.events_text.see('1.0')


# ---------------------------------------------------------------------------
# Main Replay Viewer window
# ---------------------------------------------------------------------------

class ReplayViewer:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.replay: dict | None = None
        self.current_turn: int = 0
        self.autoplay_id = None
        self.speed_var = tk.DoubleVar(value=1.0)

        root.title('Pokejax Replay Viewer')
        root.configure(bg=BG)
        root.geometry('1300x850')
        root.minsize(900, 600)

        self._build_ui()
        self._bind_keys()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ----- Control bar -----
        ctrl = tk.Frame(self.root, bg=BG_PANEL, pady=6)
        ctrl.pack(side='top', fill='x', padx=0, pady=0)

        self.lbl_file = tk.Label(ctrl, text='No replay loaded', bg=BG_PANEL,
                                 fg=FG_DIM, font=FONT_SMALL, anchor='w')
        self.lbl_file.pack(side='left', padx=8)

        btn_open = tk.Button(ctrl, text='Open Replay', command=self._open_file,
                             bg=ACCENT, fg=FG, font=FONT_LABEL,
                             relief='flat', padx=8, cursor='hand2')
        btn_open.pack(side='left', padx=4)

        ttk.Separator(ctrl, orient='vertical').pack(side='left', padx=10, fill='y')

        self.btn_first = tk.Button(ctrl, text='|◀', command=self._go_first,
                                   bg=BG_CARD, fg=FG, font=FONT_LABEL,
                                   relief='flat', padx=6, cursor='hand2')
        self.btn_first.pack(side='left', padx=2)

        self.btn_prev = tk.Button(ctrl, text='◀', command=self._go_prev,
                                  bg=BG_CARD, fg=FG, font=FONT_LABEL,
                                  relief='flat', padx=6, cursor='hand2')
        self.btn_prev.pack(side='left', padx=2)

        self.lbl_turn = tk.Label(ctrl, text='— / —', bg=BG_PANEL, fg=FG,
                                 font=('Consolas', 11, 'bold'), width=10)
        self.lbl_turn.pack(side='left', padx=8)

        self.btn_next = tk.Button(ctrl, text='▶', command=self._go_next,
                                  bg=BG_CARD, fg=FG, font=FONT_LABEL,
                                  relief='flat', padx=6, cursor='hand2')
        self.btn_next.pack(side='left', padx=2)

        self.btn_last = tk.Button(ctrl, text='▶|', command=self._go_last,
                                  bg=BG_CARD, fg=FG, font=FONT_LABEL,
                                  relief='flat', padx=6, cursor='hand2')
        self.btn_last.pack(side='left', padx=2)

        ttk.Separator(ctrl, orient='vertical').pack(side='left', padx=10, fill='y')

        self.btn_play = tk.Button(ctrl, text='▶ Auto', command=self._toggle_autoplay,
                                  bg='#225522', fg=FG, font=FONT_LABEL,
                                  relief='flat', padx=8, cursor='hand2')
        self.btn_play.pack(side='left', padx=2)

        tk.Label(ctrl, text='Speed:', bg=BG_PANEL, fg=FG_DIM,
                 font=FONT_SMALL).pack(side='left', padx=(10, 2))
        speed_scale = ttk.Scale(ctrl, from_=0.25, to=5.0,
                                variable=self.speed_var, orient='horizontal',
                                length=100)
        speed_scale.pack(side='left', padx=2)
        self.lbl_speed = tk.Label(ctrl, text='1.0×', bg=BG_PANEL, fg=FG_DIM,
                                  font=FONT_SMALL, width=4)
        self.lbl_speed.pack(side='left', padx=2)
        self.speed_var.trace_add('write', self._on_speed_change)

        # Turn slider
        ttk.Separator(ctrl, orient='vertical').pack(side='left', padx=10, fill='y')
        self.turn_var = tk.IntVar(value=0)
        self.slider = ttk.Scale(ctrl, from_=0, to=0, variable=self.turn_var,
                                orient='horizontal', length=200,
                                command=self._on_slider)
        self.slider.pack(side='left', padx=4)

        # Result label (right side)
        self.lbl_result = tk.Label(ctrl, text='', bg=BG_PANEL, fg='#aaff88',
                                   font=FONT_TITLE)
        self.lbl_result.pack(side='right', padx=12)

        # ----- Main content area -----
        main = tk.Frame(self.root, bg=BG)
        main.pack(side='top', fill='both', expand=True, padx=4, pady=4)
        main.columnconfigure(0, weight=1, minsize=300)
        main.columnconfigure(1, weight=0, minsize=280)
        main.columnconfigure(2, weight=1, minsize=300)
        main.rowconfigure(0, weight=1)

        # P1 team panel
        self.panel_p1 = TeamPanel(main, 'P1')
        self.panel_p1.frame.grid(row=0, column=0, sticky='nsew', padx=(0, 2))

        # Center: field + events
        self.center = CenterPanel(main)
        self.center.frame.grid(row=0, column=1, sticky='nsew', padx=2)

        # P2 team panel
        self.panel_p2 = TeamPanel(main, 'P2')
        self.panel_p2.frame.grid(row=0, column=2, sticky='nsew', padx=(2, 0))

        # ----- Status bar -----
        self.status_bar = tk.Label(self.root, text='Ready. Open a replay file to begin.',
                                   bg=BG_PANEL, fg=FG_DIM, font=FONT_SMALL,
                                   anchor='w', relief='flat', pady=3)
        self.status_bar.pack(side='bottom', fill='x', padx=4, pady=(2, 0))

    def _bind_keys(self):
        self.root.bind('<Left>',  lambda e: self._go_prev())
        self.root.bind('<Right>', lambda e: self._go_next())
        self.root.bind('<Home>',  lambda e: self._go_first())
        self.root.bind('<End>',   lambda e: self._go_last())
        self.root.bind('<space>', lambda e: self._toggle_autoplay())
        self.root.bind('<o>',     lambda e: self._open_file())

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def _open_file(self):
        path = filedialog.askopenfilename(
            title='Open Replay File',
            filetypes=[('Replay files', '*.pkl'), ('All files', '*.*')],
            initialdir=os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), 'replays'),
        )
        if path:
            self.load_replay(path)

    def load_replay(self, path: str):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            messagebox.showerror('Error', f'Failed to load replay:\n{e}')
            return

        # Validate basic structure
        if 'turns' not in data or 'tables' not in data:
            messagebox.showerror('Error', 'Invalid replay file format.')
            return

        self.replay = data
        self.current_turn = 0

        # Update slider range (n = last turn, n = final frame)
        n = len(data['turns'])
        self.slider.configure(to=max(n, 0))
        self.turn_var.set(0)

        # File name label
        fname = os.path.basename(path)
        meta = data.get('metadata', {})
        p0 = meta.get('p0_name', 'P1')
        p1 = meta.get('p1_name', 'P2')
        gen = meta.get('gen', '?')
        self.lbl_file.configure(
            text=f'{fname}  |  Gen {gen}  |  {p0} vs {p1}')

        # Result
        result = data.get('result', '?')
        self.lbl_result.configure(text=result)

        # Status bar
        self.status_bar.configure(
            text=f'Loaded {n} turns  •  {p0} vs {p1}  •  Result: {result}')

        self._refresh_turn()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _go_to(self, n: int):
        if self.replay is None:
            return
        # Index n == len(turns) means "show final state after last turn"
        n = max(0, min(n, len(self.replay['turns'])))
        self.current_turn = n
        self.turn_var.set(min(n, len(self.replay['turns']) - 1))
        self._refresh_turn()

    def _go_first(self): self._go_to(0)
    def _go_last(self):  self._go_to(len(self.replay['turns']) if self.replay else 0)
    def _go_prev(self):  self._go_to(self.current_turn - 1)
    def _go_next(self):  self._go_to(self.current_turn + 1)

    def _on_slider(self, val):
        if self.replay is None:
            return
        n = int(float(val))
        if n != self.current_turn:
            self.current_turn = n
            self._refresh_turn()

    # ------------------------------------------------------------------
    # Auto-play
    # ------------------------------------------------------------------

    def _toggle_autoplay(self):
        if self.autoplay_id is not None:
            self.root.after_cancel(self.autoplay_id)
            self.autoplay_id = None
            self.btn_play.configure(text='▶ Auto', bg='#225522')
        else:
            if self.replay is None:
                return
            self.btn_play.configure(text='⏸ Pause', bg='#552222')
            # Use a sentinel so _autoplay_step knows it's running
            self.autoplay_id = 'running'
            self._autoplay_step()

    def _autoplay_step(self):
        if self.autoplay_id is None:
            return
        if self.replay is None or self.current_turn >= len(self.replay['turns']) - 1:
            self._toggle_autoplay()
            return
        self._go_next()
        speed = self.speed_var.get()
        delay_ms = max(50, int(1000 / max(speed, 0.01)))
        self.autoplay_id = self.root.after(delay_ms, self._autoplay_step)

    def _on_speed_change(self, *_):
        v = self.speed_var.get()
        self.lbl_speed.configure(text=f'{v:.1f}×')

    # ------------------------------------------------------------------
    # Display refresh
    # ------------------------------------------------------------------

    def _refresh_turn(self):
        if self.replay is None:
            return

        turns = self.replay['turns']
        tables = self.replay['tables']
        meta   = self.replay.get('metadata', {})
        p0     = meta.get('p0_name', 'P1')
        p1     = meta.get('p1_name', 'P2')
        n      = len(turns)
        i      = self.current_turn

        # Turn label — show "state before" or "final state"
        if i < n:
            turn_data = turns[i]
            snap   = turn_data['state']
            events = turn_data['events']
            game_turn = int(snap.get('turn', i + 1))
            self.lbl_turn.configure(text=f'Turn {game_turn}  ({i + 1}/{n})')
        else:
            # "Final" frame: show state after the last turn
            turn_data = turns[-1]
            snap   = turn_data.get('state_after', turn_data['state'])
            events = [('field', f'⚔ Battle ended: {self.replay.get("result", "?")}')]
            game_turn = int(snap.get('turn', n))
            self.lbl_turn.configure(text=f'Final  ({n}/{n})')

        self.panel_p1.update(snap, 0, tables, p0)
        self.panel_p2.update(snap, 1, tables, p1)
        self.center.update_field(snap, [p0, p1])
        self.center.update_events(events)

        # Status bar
        alive0 = int(snap['pokemon_left'][0])
        alive1 = int(snap['pokemon_left'][1])
        self.status_bar.configure(
            text=(f'Turn {game_turn}  •  {p0}: {alive0}/6 alive  •  '
                  f'{p1}: {alive1}/6 alive  •  '
                  f'Result: {self.replay.get("result", "?")}'))

    # ------------------------------------------------------------------
    # Speed callback placeholder (already done via trace_add)
    # ------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Pokejax Battle Replay Viewer')
    parser.add_argument('replay', nargs='?', default=None,
                        help='Path to replay .pkl file (optional, can open via GUI)')
    args = parser.parse_args()

    root = tk.Tk()

    # Apply ttk dark theme if available
    style = ttk.Style(root)
    available = style.theme_names()
    if 'clam' in available:
        style.theme_use('clam')
    style.configure('TScale', background=BG_PANEL)
    style.configure('TScrollbar', background=BG_CARD, troughcolor=BG)

    viewer = ReplayViewer(root)

    if args.replay:
        if os.path.exists(args.replay):
            viewer.load_replay(args.replay)
        else:
            messagebox.showerror('Error', f'File not found: {args.replay}')

    root.mainloop()


if __name__ == '__main__':
    main()
