import { createContext } from "react";

export type Player = 'red' | 'blue';

export type Dot = {
  player: Player;
};

export type Polygon = {
  id: number;
  points: Array<[i: number, j: number]>;
  player: Player;
};

export type GameState = {
  occupied_spaces: Set<[i: number, j: number]>;
  dots: Map<[i: number, j: number], Player>;
  polygons: Array<Polygon>;
};

export const GameContext = createContext<GameState | undefined>(undefined);
