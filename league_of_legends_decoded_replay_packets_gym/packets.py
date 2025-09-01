#!/usr/bin/env python3
"""
League of Legends Packet Definitions

This module contains comprehensive definitions for all League of Legends replay packets
as documented in the Rust implementation. These packets represent all possible game events
that can occur during a League of Legends match.
"""

from dataclasses import dataclass
from typing import Dict, List, Union, Optional, Any
from enum import Enum
import json

from .types import Position


class AIType(Enum):
    """AI type classification for game entities"""
    UNKNOWN = "Unknown"
    HERO = "Hero"
    MINION = "Minion"
    TURRET = "Turret"
    NEUTRAL = "Neutral"


class ReplicationInternalData:
    """Base class for replication data values"""
    pass


@dataclass
class IntReplicationData(ReplicationInternalData):
    value: int


@dataclass
class FloatReplicationData(ReplicationInternalData):
    value: float


@dataclass
class ReplicationData:
    """Replication data structure for game state synchronization"""
    primary_index: int
    secondary_index: int
    name: str
    data: ReplicationInternalData


# Packet Definitions - Each packet represents a specific game event

@dataclass
class CreateHero:
    """Creates a new hero in the game"""
    time: float
    net_id: int
    name: str
    champion: str


@dataclass
class WaypointGroup:
    """Movement waypoints for entities"""
    time: float
    waypoints: Dict[int, List[Position]]  # net_id -> list of positions


@dataclass
class WaypointGroupWithSpeed:
    """Movement waypoints with speed information"""
    time: float
    waypoints: Dict[int, List[Position]]  # net_id -> list of positions


@dataclass
class EnterFog:
    """Entity enters fog of war"""
    time: float
    net_id: int


@dataclass
class LeaveFog:
    """Entity leaves fog of war"""
    time: float
    net_id: int


@dataclass
class UnitApplyDamage:
    """Damage application between units"""
    time: float
    source_net_id: int
    target_net_id: int
    damage: float


@dataclass
class DoSetCooldown:
    """Sets ability cooldown"""
    time: float
    net_id: int
    slot: int
    cooldown: float
    display_cooldown: float


@dataclass
class BasicAttackPos:
    """Basic attack with positional information"""
    time: float
    source_net_id: int
    target_net_id: int
    source_position: Position
    target_position: Position
    slot: int
    caster_net_id: int
    spell_chain_owner_net_id: int
    spell_hash: int
    spell_name: str
    level: int
    target_end_position: Position
    target_net_ids: List[int]
    windup_time: float
    cooldown: float
    mana_cost: float


@dataclass
class CastSpellAns:
    """Spell casting event with complete information"""
    time: float
    caster_net_id: int
    spell_chain_owner_net_id: int
    spell_hash: int
    spell_name: str
    level: int
    source_position: Position
    target_position: Position
    target_end_position: Position
    target_net_ids: List[int]
    windup_time: float
    cooldown: float
    mana_cost: float
    slot: int


@dataclass
class BarrackSpawnUnit:
    """Barrack spawns a unit (minions)"""
    time: float
    barrack_net_id: int
    minion_net_id: int
    wave_count: int
    minion_type: int  # 0xc0 = cannon, 0x60 = ranged, 0 = melee
    minion_level: int


@dataclass
class Replication:
    """Game state replication data"""
    time: float
    net_id_to_replication_datas: Dict[int, ReplicationData]


@dataclass
class SpawnMinion:
    """Spawns a minion unit"""
    time: float
    net_id: int
    position1: Position
    position2: Position
    name: str
    skin_name: str
    level: int
    targetable_on_client: int
    targetable_to_team_flags_on_client: int
    bot: bool


@dataclass
class CreateNeutral:
    """Creates neutral monster/jungle creep"""
    time: float
    net_id: int
    position1: Position
    position2: Position
    name: str
    skin_name: str
    level: int
    direction: Position
    camp_id: int
    neutral_type: int


@dataclass
class CreateTurret:
    """Creates a turret structure"""
    time: float
    net_id: int
    owner_net_id: int
    name: str


@dataclass
class NPCDieMapView:
    """NPC death event for map view"""
    time: float
    killer_net_id: int
    killed_net_id: int


@dataclass
class NPCDieMapViewBroadcast:
    """Broadcasted NPC death event"""
    time: float
    killer_net_id: int
    killed_net_id: int


@dataclass
class HeroDie:
    """Hero death event"""
    time: float
    net_id: int


@dataclass
class BuyItem:
    """Item purchase event"""
    time: float
    net_id: int
    slot: int
    item_id: int
    item_name: str
    items_in_slot: int
    spell_charges: int
    item_gold: float
    entity_gold_after_change: float


@dataclass
class RemoveItem:
    """Item removal/sell event"""
    time: float
    net_id: int
    slot: int
    items_in_slot: int
    entity_gold_after_change: float


@dataclass
class SwapItem:
    """Item slot swap event"""
    time: float
    net_id: int
    source_slot: int
    target_slot: int


@dataclass
class UseItem:
    """Item usage event"""
    time: float
    net_id: int
    slot: int
    items_in_slot: int
    spell_charges: int


# Union type for all possible packets
PacketType = Union[
    CreateHero,
    WaypointGroup,
    WaypointGroupWithSpeed,
    EnterFog,
    LeaveFog,
    UnitApplyDamage,
    DoSetCooldown,
    BasicAttackPos,
    CastSpellAns,
    BarrackSpawnUnit,
    Replication,
    SpawnMinion,
    CreateNeutral,
    CreateTurret,
    NPCDieMapView,
    NPCDieMapViewBroadcast,
    HeroDie,
    BuyItem,
    RemoveItem,
    SwapItem,
    UseItem,
]


class PacketParser:
    """Parser for converting raw JSON packet data into typed packet objects"""
    
    @staticmethod
    def parse_position(data: Dict[str, float]) -> Position:
        """Parse position from JSON data"""
        return Position(x=data.get("x", 0.0), z=data.get("z", 0.0))
    
    @staticmethod
    def parse_waypoints(waypoints_data: Dict[str, List[Dict[str, float]]]) -> Dict[int, List[Position]]:
        """Parse waypoints dictionary"""
        waypoints = {}
        for net_id_str, positions in waypoints_data.items():
            net_id = int(net_id_str)
            waypoints[net_id] = [
                PacketParser.parse_position(pos) for pos in positions
            ]
        return waypoints
    
    @staticmethod
    def parse_replication_data(data: Dict[str, Any]) -> ReplicationData:
        """Parse replication data"""
        internal_data = data.get("data")
        if isinstance(internal_data, dict):
            if "Int" in internal_data:
                repl_data = IntReplicationData(internal_data["Int"])
            elif "Float" in internal_data:
                repl_data = FloatReplicationData(internal_data["Float"])
            else:
                repl_data = IntReplicationData(0)  # Default fallback
        else:
            repl_data = IntReplicationData(0)  # Default fallback
        
        return ReplicationData(
            primary_index=data.get("primary_index", 0),
            secondary_index=data.get("secondary_index", 0),
            name=data.get("name", ""),
            data=repl_data
        )
    
    @classmethod
    def parse_packet(cls, packet_data: Dict[str, Any]) -> Optional[PacketType]:
        """Parse a packet from raw JSON data"""
        if not packet_data:
            return None
        
        # Get packet type and data
        packet_type = list(packet_data.keys())[0]
        data = packet_data[packet_type]
        
        try:
            if packet_type == "CreateHero":
                return CreateHero(
                    time=data["time"],
                    net_id=data["net_id"],
                    name=data["name"],
                    champion=data["champion"]
                )
            
            elif packet_type == "WaypointGroup":
                return WaypointGroup(
                    time=data["time"],
                    waypoints=cls.parse_waypoints(data["waypoints"])
                )
            
            elif packet_type == "WaypointGroupWithSpeed":
                return WaypointGroupWithSpeed(
                    time=data["time"],
                    waypoints=cls.parse_waypoints(data["waypoints"])
                )
            
            elif packet_type == "EnterFog":
                return EnterFog(
                    time=data["time"],
                    net_id=data["net_id"]
                )
            
            elif packet_type == "LeaveFog":
                return LeaveFog(
                    time=data["time"],
                    net_id=data["net_id"]
                )
            
            elif packet_type == "UnitApplyDamage":
                return UnitApplyDamage(
                    time=data["time"],
                    source_net_id=data["source_net_id"],
                    target_net_id=data["target_net_id"],
                    damage=data["damage"]
                )
            
            elif packet_type == "DoSetCooldown":
                return DoSetCooldown(
                    time=data["time"],
                    net_id=data["net_id"],
                    slot=data["slot"],
                    cooldown=data["cooldown"],
                    display_cooldown=data["display_cooldown"]
                )
            
            elif packet_type == "BasicAttackPos":
                return BasicAttackPos(
                    time=data["time"],
                    source_net_id=data["source_net_id"],
                    target_net_id=data["target_net_id"],
                    source_position=cls.parse_position(data["source_position"]),
                    target_position=cls.parse_position(data["target_position"]),
                    slot=data["slot"],
                    caster_net_id=data["caster_net_id"],
                    spell_chain_owner_net_id=data["spell_chain_owner_net_id"],
                    spell_hash=data["spell_hash"],
                    spell_name=data["spell_name"],
                    level=data["level"],
                    target_end_position=cls.parse_position(data["target_end_position"]),
                    target_net_ids=data["target_net_ids"],
                    windup_time=data["windup_time"],
                    cooldown=data["cooldown"],
                    mana_cost=data["mana_cost"]
                )
            
            elif packet_type == "CastSpellAns":
                return CastSpellAns(
                    time=data["time"],
                    caster_net_id=data["caster_net_id"],
                    spell_chain_owner_net_id=data["spell_chain_owner_net_id"],
                    spell_hash=data["spell_hash"],
                    spell_name=data["spell_name"],
                    level=data["level"],
                    source_position=cls.parse_position(data["source_position"]),
                    target_position=cls.parse_position(data["target_position"]),
                    target_end_position=cls.parse_position(data["target_end_position"]),
                    target_net_ids=data["target_net_ids"],
                    windup_time=data["windup_time"],
                    cooldown=data["cooldown"],
                    mana_cost=data["mana_cost"],
                    slot=data["slot"]
                )
            
            elif packet_type == "BarrackSpawnUnit":
                return BarrackSpawnUnit(
                    time=data["time"],
                    barrack_net_id=data["barrack_net_id"],
                    minion_net_id=data["minion_net_id"],
                    wave_count=data["wave_count"],
                    minion_type=data["minion_type"],
                    minion_level=data["minion_level"]
                )
            
            elif packet_type == "Replication":
                replication_datas = {}
                for net_id_str, repl_data in data["net_id_to_replication_datas"].items():
                    net_id = int(net_id_str)
                    replication_datas[net_id] = cls.parse_replication_data(repl_data)
                
                return Replication(
                    time=data["time"],
                    net_id_to_replication_datas=replication_datas
                )
            
            elif packet_type == "SpawnMinion":
                return SpawnMinion(
                    time=data["time"],
                    net_id=data["net_id"],
                    position1=cls.parse_position(data["position1"]),
                    position2=cls.parse_position(data["position2"]),
                    name=data["name"],
                    skin_name=data["skin_name"],
                    level=data["level"],
                    targetable_on_client=data["targetable_on_client"],
                    targetable_to_team_flags_on_client=data["targetable_to_team_flags_on_client"],
                    bot=data["bot"]
                )
            
            elif packet_type == "CreateNeutral":
                return CreateNeutral(
                    time=data["time"],
                    net_id=data["net_id"],
                    position1=cls.parse_position(data["position1"]),
                    position2=cls.parse_position(data["position2"]),
                    name=data["name"],
                    skin_name=data["skin_name"],
                    level=data["level"],
                    direction=cls.parse_position(data["direction"]),
                    camp_id=data["camp_id"],
                    neutral_type=data["neutral_type"]
                )
            
            elif packet_type == "CreateTurret":
                return CreateTurret(
                    time=data["time"],
                    net_id=data["net_id"],
                    owner_net_id=data["owner_net_id"],
                    name=data["name"]
                )
            
            elif packet_type == "NPCDieMapView":
                return NPCDieMapView(
                    time=data["time"],
                    killer_net_id=data["killer_net_id"],
                    killed_net_id=data["killed_net_id"]
                )
            
            elif packet_type == "NPCDieMapViewBroadcast":
                return NPCDieMapViewBroadcast(
                    time=data["time"],
                    killer_net_id=data["killer_net_id"],
                    killed_net_id=data["killed_net_id"]
                )
            
            elif packet_type == "HeroDie":
                return HeroDie(
                    time=data["time"],
                    net_id=data["net_id"]
                )
            
            elif packet_type == "BuyItem":
                return BuyItem(
                    time=data["time"],
                    net_id=data["net_id"],
                    slot=data["slot"],
                    item_id=data["item_id"],
                    item_name=data["item_name"],
                    items_in_slot=data["items_in_slot"],
                    spell_charges=data["spell_charges"],
                    item_gold=data["item_gold"],
                    entity_gold_after_change=data["entity_gold_after_change"]
                )
            
            elif packet_type == "RemoveItem":
                return RemoveItem(
                    time=data["time"],
                    net_id=data["net_id"],
                    slot=data["slot"],
                    items_in_slot=data["items_in_slot"],
                    entity_gold_after_change=data["entity_gold_after_change"]
                )
            
            elif packet_type == "SwapItem":
                return SwapItem(
                    time=data["time"],
                    net_id=data["net_id"],
                    source_slot=data["source_slot"],
                    target_slot=data["target_slot"]
                )
            
            elif packet_type == "UseItem":
                return UseItem(
                    time=data["time"],
                    net_id=data["net_id"],
                    slot=data["slot"],
                    items_in_slot=data["items_in_slot"],
                    spell_charges=data["spell_charges"]
                )
            
            else:
                print(f"Warning: Unknown packet type: {packet_type}")
                return None
        
        except KeyError as e:
            print(f"Error parsing {packet_type}: missing field {e}")
            return None
        except Exception as e:
            print(f"Error parsing {packet_type}: {e}")
            return None


def get_packet_types() -> List[str]:
    """Get list of all supported packet types"""
    return [
        "CreateHero",
        "WaypointGroup", 
        "WaypointGroupWithSpeed",
        "EnterFog",
        "LeaveFog",
        "UnitApplyDamage",
        "DoSetCooldown",
        "BasicAttackPos",
        "CastSpellAns",
        "BarrackSpawnUnit",
        "Replication",
        "SpawnMinion",
        "CreateNeutral",
        "CreateTurret",
        "NPCDieMapView",
        "NPCDieMapViewBroadcast",
        "HeroDie",
        "BuyItem",
        "RemoveItem",
        "SwapItem",
        "UseItem"
    ]


def get_packet_schema() -> Dict[str, Dict[str, Any]]:
    """Get schema description for all packet types"""
    return {
        "CreateHero": {
            "description": "Creates a new hero in the game",
            "fields": ["time", "net_id", "name", "champion"]
        },
        "WaypointGroup": {
            "description": "Movement waypoints for entities",
            "fields": ["time", "waypoints"]
        },
        "WaypointGroupWithSpeed": {
            "description": "Movement waypoints with speed information", 
            "fields": ["time", "waypoints"]
        },
        "EnterFog": {
            "description": "Entity enters fog of war",
            "fields": ["time", "net_id"]
        },
        "LeaveFog": {
            "description": "Entity leaves fog of war",
            "fields": ["time", "net_id"]
        },
        "UnitApplyDamage": {
            "description": "Damage application between units",
            "fields": ["time", "source_net_id", "target_net_id", "damage"]
        },
        "DoSetCooldown": {
            "description": "Sets ability cooldown",
            "fields": ["time", "net_id", "slot", "cooldown", "display_cooldown"]
        },
        "BasicAttackPos": {
            "description": "Basic attack with positional information",
            "fields": ["time", "source_net_id", "target_net_id", "source_position", "target_position", "slot", "caster_net_id", "spell_chain_owner_net_id", "spell_hash", "spell_name", "level", "target_end_position", "target_net_ids", "windup_time", "cooldown", "mana_cost"]
        },
        "CastSpellAns": {
            "description": "Spell casting event with complete information",
            "fields": ["time", "caster_net_id", "spell_chain_owner_net_id", "spell_hash", "spell_name", "level", "source_position", "target_position", "target_end_position", "target_net_ids", "windup_time", "cooldown", "mana_cost", "slot"]
        },
        "BarrackSpawnUnit": {
            "description": "Barrack spawns a unit (minions)",
            "fields": ["time", "barrack_net_id", "minion_net_id", "wave_count", "minion_type", "minion_level"]
        },
        "Replication": {
            "description": "Game state replication data",
            "fields": ["time", "net_id_to_replication_datas"]
        },
        "SpawnMinion": {
            "description": "Spawns a minion unit",
            "fields": ["time", "net_id", "position1", "position2", "name", "skin_name", "level", "targetable_on_client", "targetable_to_team_flags_on_client", "bot"]
        },
        "CreateNeutral": {
            "description": "Creates neutral monster/jungle creep",
            "fields": ["time", "net_id", "position1", "position2", "name", "skin_name", "level", "direction", "camp_id", "neutral_type"]
        },
        "CreateTurret": {
            "description": "Creates a turret structure",
            "fields": ["time", "net_id", "owner_net_id", "name"]
        },
        "NPCDieMapView": {
            "description": "NPC death event for map view",
            "fields": ["time", "killer_net_id", "killed_net_id"]
        },
        "NPCDieMapViewBroadcast": {
            "description": "Broadcasted NPC death event",
            "fields": ["time", "killer_net_id", "killed_net_id"]
        },
        "HeroDie": {
            "description": "Hero death event",
            "fields": ["time", "net_id"]
        },
        "BuyItem": {
            "description": "Item purchase event",
            "fields": ["time", "net_id", "slot", "item_id", "item_name", "items_in_slot", "spell_charges", "item_gold", "entity_gold_after_change"]
        },
        "RemoveItem": {
            "description": "Item removal/sell event",
            "fields": ["time", "net_id", "slot", "items_in_slot", "entity_gold_after_change"]
        },
        "SwapItem": {
            "description": "Item slot swap event",
            "fields": ["time", "net_id", "source_slot", "target_slot"]
        },
        "UseItem": {
            "description": "Item usage event", 
            "fields": ["time", "net_id", "slot", "items_in_slot", "spell_charges"]
        }
    }