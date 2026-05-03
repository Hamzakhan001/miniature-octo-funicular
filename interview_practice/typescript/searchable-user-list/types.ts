export type User = {
  id: number;
  name: string;
  email: string;
  company: string;
};

export type UserListState = {
  users: User[];
  searchTerm: string;
  isLoading: boolean;
  error: string | null;
};

